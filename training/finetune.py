"""
finetune.py
-----------
Stage 2: Fine-tune the pretrained GPT on DailyDialog conversations.

What changes vs pretraining
----------------------------
1. Lower learning rate (3e-5 vs 3e-4) — we want small updates that
   specialise the pretrained weights, not overwrite them.  Training with
   a high LR here causes "catastrophic forgetting" — the model loses its
   language knowledge and just memorises dialogue patterns.

2. Masked loss — we only backpropagate on ASSISTANT tokens (the replies),
   not on HUMAN tokens (the prompts).  This focuses all gradient signal
   on learning how to respond, not on learning to predict questions.

3. Vocabulary expansion — DailyDialog tokenizer has 3 extra special
   tokens (<human>, <assistant>, </s>) added on top of GPT-2's 50,257.
   The embedding and lm_head are resized to accommodate them, with the
   original pretrained weights preserved.

Usage:
    python -m training.finetune                          # from best pretrain checkpoint
    python -m training.finetune --pretrain path/to.pt   # custom checkpoint
    python -m training.finetune --resume                 # continue a finetune run
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.gpt import GPT
from dataset.dialog_dataset import make_dialog_datasets

from configs.config import (
    PRETRAIN_BLOCK_SIZE,
    PRETRAIN_D_MODEL, PRETRAIN_NUM_HEADS, PRETRAIN_NUM_LAYERS, PRETRAIN_DROPOUT,
    FINETUNE_BLOCK_SIZE,
    FINETUNE_BATCH_SIZE, FINETUNE_LEARNING_RATE,
    FINETUNE_WEIGHT_DECAY, FINETUNE_EPOCHS, FINETUNE_STEPS_PER_EPOCH,
    PRETRAIN_CKPT, FINETUNE_CKPT, CHECKPOINT_DIR,
    SEED, DEVICE,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pretrain", type=str, default=str(PRETRAIN_CKPT),
                    help="Path to pretrained checkpoint (default: pretrain_best.pt)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from a previous finetune checkpoint")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)

print(f"Device       : {DEVICE}")
print(f"Pretrain ckpt: {args.pretrain}")
print(f"Training     : batch={FINETUNE_BATCH_SIZE}, epochs={FINETUNE_EPOCHS}, "
      f"steps/epoch={FINETUNE_STEPS_PER_EPOCH}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train_ds, val_ds, vocab_size, tokenizer = make_dialog_datasets(FINETUNE_BLOCK_SIZE)

train_loader = DataLoader(
    train_ds, batch_size=FINETUNE_BATCH_SIZE,
    shuffle=True, num_workers=0,
)
val_loader = DataLoader(
    val_ds, batch_size=FINETUNE_BATCH_SIZE,
    shuffle=False, num_workers=0,
)

val_steps = max(1, FINETUNE_STEPS_PER_EPOCH // 5)
print(f"Vocab size (with special tokens): {vocab_size:,}")

# ---------------------------------------------------------------------------
# Model — load pretrained weights then resize for new vocab
# ---------------------------------------------------------------------------
if not os.path.exists(args.pretrain):
    raise FileNotFoundError(
        f"Pretrain checkpoint not found: {args.pretrain}\n"
        f"Run `python -m training.pretrain` first."
    )

print(f"Loading pretrained model …")
ckpt = torch.load(args.pretrain, map_location=DEVICE)
pretrain_cfg = ckpt["config"]

# Build the model with the PRETRAINED vocab size first
model = GPT(
    vocab_size  = pretrain_cfg["vocab_size"],
    max_seq_len = pretrain_cfg["max_seq_len"],
    d_model     = pretrain_cfg["d_model"],
    num_heads   = pretrain_cfg["num_heads"],
    num_layers  = pretrain_cfg["num_layers"],
    dropout     = pretrain_cfg["dropout"],
)
model.load_state_dict(ckpt["model"])
print(f"  Loaded pretrain epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f}")

# Resize embedding + lm_head for the expanded vocabulary.
# New token rows are initialised from the mean of existing embeddings —
# a better starting point than random init, giving them a reasonable
# position in the embedding space right away.
old_vocab = pretrain_cfg["vocab_size"]
new_vocab = vocab_size

if new_vocab != old_vocab:
    print(f"  Resizing vocab {old_vocab:,} → {new_vocab:,} …")

    old_emb = model.embedding.token_embedding.weight.data  # (old_vocab, d)
    mean_emb = old_emb.mean(dim=0, keepdim=True)           # (1, d)

    new_emb = nn.Embedding(new_vocab, pretrain_cfg["d_model"])
    new_emb.weight.data[:old_vocab] = old_emb
    new_emb.weight.data[old_vocab:] = mean_emb.expand(new_vocab - old_vocab, -1)
    model.embedding.token_embedding = new_emb

    # lm_head shares weights with token_embedding (weight tying)
    model.lm_head = nn.Linear(pretrain_cfg["d_model"], new_vocab, bias=False)
    model.lm_head.weight = model.embedding.token_embedding.weight

model = model.to(DEVICE)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {num_params:,}")

# ---------------------------------------------------------------------------
# Optimiser + scheduler
# ---------------------------------------------------------------------------
# Much lower LR than pretraining — small nudges only.
# We also use a shorter cosine cycle so the LR drops to near-zero by the
# end of fine-tuning, letting the model "settle" its new dialogue weights.
decay_p    = [p for n, p in model.named_parameters()
              if p.requires_grad and p.dim() >= 2]
no_decay_p = [p for n, p in model.named_parameters()
              if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW(
    [{"params": decay_p,    "weight_decay": FINETUNE_WEIGHT_DECAY},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=FINETUNE_LEARNING_RATE,
    betas=(0.9, 0.95),
)

total_steps  = FINETUNE_STEPS_PER_EPOCH * FINETUNE_EPOCHS
warmup_steps = max(1, total_steps // 20)   # 5% warmup (shorter than pretrain)

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Masked cross-entropy: `reduction="none"` gives per-token loss,
# which we then multiply by the mask and average over non-zero positions.
# This means only assistant tokens contribute to the gradient.
criterion = nn.CrossEntropyLoss(reduction="none")

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
start_epoch   = 1
best_val_loss = float("inf")

if args.resume and os.path.exists(str(FINETUNE_CKPT)):
    print(f"Resuming fine-tune from {FINETUNE_CKPT} …")
    resume_ckpt = torch.load(str(FINETUNE_CKPT), map_location=DEVICE)
    model.load_state_dict(resume_ckpt["model"])
    optimizer.load_state_dict(resume_ckpt["optimizer"])
    scheduler.load_state_dict(resume_ckpt["scheduler"])
    start_epoch   = resume_ckpt["epoch"] + 1
    best_val_loss = resume_ckpt["val_loss"]
    print(f"  Resumed at epoch {start_epoch}")

# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------
def run_epoch(loader, training: bool, epoch: int, max_steps: int) -> float:
    model.train() if training else model.eval()

    total_loss = 0.0
    label      = "train" if training else "val  "
    loader_it  = iter(loader)
    step       = 0

    bar = tqdm(
        total=max_steps,
        desc=f"Epoch {epoch:>2}/{FINETUNE_EPOCHS} {label}",
        unit="batch",
        leave=training,
        dynamic_ncols=True,
    )

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        while step < max_steps:
            try:
                x, y, mask = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                x, y, mask = next(loader_it)

            x    = x.to(DEVICE)           # (B, T)
            y    = y.to(DEVICE)           # (B, T)
            mask = mask.to(DEVICE)        # (B, T) — 1.0 on assistant tokens

            logits   = model(x)           # (B, T, vocab_size)
            B, T, V  = logits.shape

            # Per-token loss (B*T,), then reshape back to (B, T)
            token_loss = criterion(logits.view(B * T, V), y.view(B * T))
            token_loss = token_loss.view(B, T)

            # Apply mask: zero out loss on human / padding tokens.
            # Divide by the number of unmasked tokens (not total tokens)
            # so the loss scale stays consistent regardless of how many
            # assistant tokens appear in the batch.
            masked_loss = (token_loss * mask).sum() / mask.sum().clamp(min=1)

            if training:
                optimizer.zero_grad()
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += masked_loss.item()
            step       += 1

            bar.set_postfix(
                loss=f"{total_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}" if training else "—",
            )
            bar.update(1)

    bar.close()
    return total_loss / max_steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
train_start = time.time()

for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
    t0 = time.time()

    train_loss = run_epoch(train_loader, training=True,  epoch=epoch, max_steps=FINETUNE_STEPS_PER_EPOCH)
    val_loss   = run_epoch(val_loader,   training=False, epoch=epoch, max_steps=val_steps)

    elapsed   = time.time() - t0
    train_ppl = math.exp(min(train_loss, 20))
    val_ppl   = math.exp(min(val_loss,   20))
    cur_lr    = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:>2}/{FINETUNE_EPOCHS} | "
        f"train {train_loss:.4f} (ppl {train_ppl:6.1f}) | "
        f"val {val_loss:.4f} (ppl {val_ppl:6.1f}) | "
        f"lr {cur_lr:.2e} | {elapsed:.1f}s"
    )

    ckpt_data = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "config": {
            "vocab_size":  vocab_size,
            "max_seq_len": FINETUNE_BLOCK_SIZE,
            "d_model":     pretrain_cfg["d_model"],
            "num_heads":   pretrain_cfg["num_heads"],
            "num_layers":  pretrain_cfg["num_layers"],
            "dropout":     pretrain_cfg["dropout"],
        },
    }
    torch.save(ckpt_data, str(FINETUNE_CKPT))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = str(CHECKPOINT_DIR / "finetune_best.pt")
        torch.save(ckpt_data, best_path)
        print(f"  ✓ best val loss — saved → {best_path}")

total = time.time() - train_start
print("=" * 65)
print(f"Fine-tuning done in {total/60:.1f} min  |  "
      f"Best val loss: {best_val_loss:.4f}")
print(f"\nNext step → chat with the model:")
print(f"  python -m inference.chat")
