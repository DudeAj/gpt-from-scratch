"""
pretrain.py
-----------
Stage 1: Pretrain a GPT-2 small model on WikiText-103.

This teaches the model general language — grammar, facts, word
relationships — before it sees any conversational data.

What this script does differently from train.py (shakespeare):
  - Uses GPT-2's tokenizer (50,257 vocab) instead of our tiny BPE
  - Model is 12 layers / d=768 (GPT-2 small, ~117M params)
  - Saves a checkpoint compatible with finetune.py

Expected training time on M3 MPS:
  ~2000 steps/epoch × 3 epochs × 115ms/step ≈ 11.5 min
  (first run also tokenises WikiText-103 — adds ~5 min one-time cost)

Usage:
    python -m training.pretrain
    python -m training.pretrain --resume   # continue from last checkpoint
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
from dataset.wikitext_dataset import make_wikitext_datasets

from configs.config import (
    PRETRAIN_BLOCK_SIZE,
    PRETRAIN_D_MODEL, PRETRAIN_NUM_HEADS, PRETRAIN_NUM_LAYERS,
    PRETRAIN_FFN_EXPANSION, PRETRAIN_DROPOUT,
    PRETRAIN_BATCH_SIZE, PRETRAIN_LEARNING_RATE,
    PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS, PRETRAIN_STEPS_PER_EPOCH,
    PRETRAIN_CKPT, CHECKPOINT_DIR,
    SEED, DEVICE,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Resume training from last pretrain checkpoint")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)

print(f"Device     : {DEVICE}")
print(f"Model      : GPT-2 small — layers={PRETRAIN_NUM_LAYERS}, "
      f"d_model={PRETRAIN_D_MODEL}, heads={PRETRAIN_NUM_HEADS}")
print(f"Training   : batch={PRETRAIN_BATCH_SIZE}, epochs={PRETRAIN_EPOCHS}, "
      f"steps/epoch={PRETRAIN_STEPS_PER_EPOCH}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# make_wikitext_datasets tokenises + caches on first call (~5 min),
# then loads from the .bin cache instantly on subsequent runs.
train_ds, val_ds, vocab_size = make_wikitext_datasets(PRETRAIN_BLOCK_SIZE)

train_loader = DataLoader(
    train_ds, batch_size=PRETRAIN_BATCH_SIZE,
    shuffle=True, num_workers=0,
)
val_loader = DataLoader(
    val_ds, batch_size=PRETRAIN_BATCH_SIZE,
    shuffle=False, num_workers=0,
)

val_steps = max(1, PRETRAIN_STEPS_PER_EPOCH // 5)
print(f"Vocab size : {vocab_size:,}")
print(f"Train examples: {len(train_ds):,}  |  Val: {len(val_ds):,}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = GPT(
    vocab_size  = vocab_size,
    max_seq_len = PRETRAIN_BLOCK_SIZE,
    d_model     = PRETRAIN_D_MODEL,
    num_heads   = PRETRAIN_NUM_HEADS,
    num_layers  = PRETRAIN_NUM_LAYERS,
    dropout     = PRETRAIN_DROPOUT,
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters : {num_params:,}  (~{num_params/1e6:.0f}M)")

# ---------------------------------------------------------------------------
# Optimiser + scheduler
# ---------------------------------------------------------------------------
# Separate weight decay groups — don't decay biases or LayerNorm params.
# For a model this size, weight decay is especially important to prevent
# overfitting on the ~500MB corpus.
decay_p    = [p for n, p in model.named_parameters()
              if p.requires_grad and p.dim() >= 2]
no_decay_p = [p for n, p in model.named_parameters()
              if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW(
    [{"params": decay_p,    "weight_decay": PRETRAIN_WEIGHT_DECAY},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=PRETRAIN_LEARNING_RATE,
    betas=(0.9, 0.95),   # GPT-2 paper values — slightly higher beta2
)

total_steps = PRETRAIN_STEPS_PER_EPOCH * PRETRAIN_EPOCHS

# Linear warmup for first 1% of steps, then cosine decay.
# Warmup prevents large gradient updates early when weights are random —
# the model would otherwise diverge on the first few batches.
warmup_steps = max(1, total_steps // 100)

def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps                         # linear ramp
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
start_epoch   = 1
best_val_loss = float("inf")

if args.resume and os.path.exists(str(PRETRAIN_CKPT)):
    print(f"Resuming from {PRETRAIN_CKPT} …")
    ckpt = torch.load(str(PRETRAIN_CKPT), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["val_loss"]
    print(f"  Resumed at epoch {start_epoch}, best val loss {best_val_loss:.4f}")

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
        desc=f"Epoch {epoch:>2}/{PRETRAIN_EPOCHS} {label}",
        unit="batch",
        leave=training,
        dynamic_ncols=True,
    )

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        while step < max_steps:
            try:
                x, y = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                x, y = next(loader_it)

            x, y = x.to(DEVICE), y.to(DEVICE)

            logits      = model(x)               # (B, T, vocab_size)
            B, T, V     = logits.shape
            loss        = criterion(logits.view(B * T, V), y.view(B * T))

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Grad clipping is critical for GPT-2 scale — without it,
                # the early steps with random weights frequently produce
                # gradient norms > 10 that blow up the optimiser state.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
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

for epoch in range(start_epoch, PRETRAIN_EPOCHS + 1):
    t0 = time.time()

    train_loss = run_epoch(train_loader, training=True,  epoch=epoch, max_steps=PRETRAIN_STEPS_PER_EPOCH)
    val_loss   = run_epoch(val_loader,   training=False, epoch=epoch, max_steps=val_steps)

    elapsed    = time.time() - t0
    train_ppl  = math.exp(min(train_loss, 20))
    val_ppl    = math.exp(min(val_loss,   20))
    cur_lr     = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:>2}/{PRETRAIN_EPOCHS} | "
        f"train {train_loss:.4f} (ppl {train_ppl:7.1f}) | "
        f"val {val_loss:.4f} (ppl {val_ppl:7.1f}) | "
        f"lr {cur_lr:.2e} | {elapsed:.1f}s"
    )

    # Always save latest; only mark as best if val improved
    ckpt_data = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "config": {
            "vocab_size":  vocab_size,
            "max_seq_len": PRETRAIN_BLOCK_SIZE,
            "d_model":     PRETRAIN_D_MODEL,
            "num_heads":   PRETRAIN_NUM_HEADS,
            "num_layers":  PRETRAIN_NUM_LAYERS,
            "dropout":     PRETRAIN_DROPOUT,
        },
    }
    torch.save(ckpt_data, str(PRETRAIN_CKPT))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = str(CHECKPOINT_DIR / "pretrain_best.pt")
        torch.save(ckpt_data, best_path)
        print(f"  ✓ best val loss — saved → {best_path}")

total = time.time() - train_start
print("=" * 65)
print(f"Pretraining done in {total/60:.1f} min  |  "
      f"Best val loss: {best_val_loss:.4f} (ppl {math.exp(min(best_val_loss, 20)):.1f})")
print(f"\nNext step → fine-tune on DailyDialog:")
print(f"  python -m training.finetune")
