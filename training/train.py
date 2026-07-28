import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tokenizer.bpe_tokenizer import BPETokenizer
from dataset.text_dataset import GPTDataset
from model.gpt import GPT

from configs.config import (
    VOCAB_SIZE, BLOCK_SIZE, TRAIN_SPLIT,
    D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS,
    STEPS_PER_EPOCH,
    SEED, DEVICE, CHECKPOINT_DIR,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)

print(f"Device     : {DEVICE}")
print(f"Model      : layers={NUM_LAYERS}, d_model={D_MODEL}, heads={NUM_HEADS}")
print(f"Training   : batch={BATCH_SIZE}, epochs={EPOCHS}, steps/epoch={STEPS_PER_EPOCH}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
TOKENIZER_PATH = str(CHECKPOINT_DIR / "tokenizer.json")

with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
if os.path.exists(TOKENIZER_PATH):
    print("Loading tokenizer from cache...")
    tokenizer.load(TOKENIZER_PATH)
else:
    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    t0 = time.time()
    tokenizer.train(text, VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Done in {time.time()-t0:.1f}s → {TOKENIZER_PATH}")

# ---------------------------------------------------------------------------
# Dataset & dataloaders
# ---------------------------------------------------------------------------
print("Encoding corpus...", end=" ", flush=True)
t0 = time.time()
tokens = tokenizer.encode(text)
print(f"{len(tokens):,} tokens ({time.time()-t0:.1f}s)")

full_dataset = GPTDataset(tokens=tokens, block_size=BLOCK_SIZE)
train_size   = int(len(full_dataset) * TRAIN_SPLIT)
val_size     = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Val steps: cap at same ratio so val is fast too
val_steps = max(1, STEPS_PER_EPOCH // 5)

print(f"Train: {len(train_dataset):,} examples → using {STEPS_PER_EPOCH} steps/epoch")
print(f"Val  : {len(val_dataset):,}  examples → using {val_steps} steps/epoch")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = GPT(
    vocab_size=VOCAB_SIZE, max_seq_len=BLOCK_SIZE,
    d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters : {num_params:,}")

# ---------------------------------------------------------------------------
# Loss & optimiser
# ---------------------------------------------------------------------------
criterion  = nn.CrossEntropyLoss()
decay_p    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
no_decay_p = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
optimizer  = torch.optim.AdamW(
    [{"params": decay_p,    "weight_decay": WEIGHT_DECAY},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=LEARNING_RATE,
)

# Cosine annealing: smoothly decays LR from LEARNING_RATE → 0 over all
# training steps. Helps the model settle into a sharper minimum at the end
# rather than bouncing around with a fixed large LR.
total_steps = STEPS_PER_EPOCH * EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=LEARNING_RATE / 10
)

# ---------------------------------------------------------------------------
# Epoch runner  (with step cap)
# ---------------------------------------------------------------------------
def run_epoch(loader, training: bool, epoch: int, max_steps: int) -> float:
    """
    Run at most `max_steps` batches from `loader`.

    Capping steps per epoch means we see a random subset of the data
    each epoch (because the train loader is shuffled). This is standard
    practice — it gives the scheduler a predictable epoch length and
    keeps wall-clock time fixed regardless of dataset size.
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    label      = "train" if training else "val  "

    bar = tqdm(
        total=max_steps,
        desc=f"Epoch {epoch:>2}/{EPOCHS} {label}",
        unit="batch",
        leave=training,
        dynamic_ncols=True,
    )

    ctx        = torch.enable_grad() if training else torch.no_grad()
    loader_it  = iter(loader)
    step       = 0

    with ctx:
        while step < max_steps:
            # Restart iterator if we exhaust the loader before max_steps
            try:
                x, y = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                x, y = next(loader_it)

            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()   # update LR after every batch

            total_loss += loss.item()
            step += 1

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
best_val_loss = float("inf")
train_start   = time.time()

print("\n" + "=" * 60)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss = run_epoch(train_loader, training=True,  epoch=epoch, max_steps=STEPS_PER_EPOCH)
    val_loss   = run_epoch(val_loader,   training=False, epoch=epoch, max_steps=val_steps)

    elapsed   = time.time() - t0
    train_ppl = math.exp(min(train_loss, 20))
    val_ppl   = math.exp(min(val_loss,   20))
    cur_lr    = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch:>2}/{EPOCHS} | "
        f"train {train_loss:.4f} (ppl {train_ppl:6.1f}) | "
        f"val {val_loss:.4f} (ppl {val_ppl:6.1f}) | "
        f"lr {cur_lr:.2e} | {elapsed:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt = str(CHECKPOINT_DIR / "best_model.pt")
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "val_loss": val_loss,
            "config": {
                "vocab_size": VOCAB_SIZE, "max_seq_len": BLOCK_SIZE,
                "d_model": D_MODEL, "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            },
        }, ckpt)
        print("  ✓ checkpoint saved (best val loss)")

total = time.time() - train_start
print("=" * 60)
print(f"Finished in {total/60:.1f} min  |  "
      f"Best val loss: {best_val_loss:.4f} (ppl {math.exp(min(best_val_loss,20)):.1f})")
