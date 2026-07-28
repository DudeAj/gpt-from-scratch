"""
prepare_data.py
---------------
Tokenize the raw text corpus once and write the token IDs to binary
files (train.bin / val.bin) so the training loop can load them instantly
without re-running the BPE tokenizer on every run.

Usage:
    python -m dataset.prepare_data

Output files (numpy uint16 arrays):
    data/train.bin
    data/val.bin
"""

import os
import numpy as np

from tokenizer.bpe_tokenizer import BPETokenizer
from configs.config import (
    TEXT_FILE,
    TRAIN_BIN,
    VAL_BIN,
    CHECKPOINT_DIR,
    VOCAB_SIZE,
    TRAIN_SPLIT,
)

# ---------------------------------------------------------------------------
# 1. Load / train tokenizer
# ---------------------------------------------------------------------------
# We reuse the same tokenizer that train.py uses so the token IDs are
# consistent. If it hasn't been trained yet we train it here.

TOKENIZER_PATH = str(CHECKPOINT_DIR / "tokenizer.json")

tokenizer = BPETokenizer()

if os.path.exists(TOKENIZER_PATH):
    print("Loading tokenizer from disk...")
    tokenizer.load(TOKENIZER_PATH)
else:
    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    with open(str(TEXT_FILE), "r", encoding="utf-8") as f:
        text = f.read()
    os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)
    tokenizer.train(text, VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved → {TOKENIZER_PATH}")

# ---------------------------------------------------------------------------
# 2. Tokenize full corpus
# ---------------------------------------------------------------------------
print(f"Reading corpus: {TEXT_FILE}")

with open(str(TEXT_FILE), "r", encoding="utf-8") as f:
    text = f.read()

print("Encoding...")
token_ids = tokenizer.encode(text)
print(f"Total tokens: {len(token_ids):,}")

# ---------------------------------------------------------------------------
# 3. Train / val split
# ---------------------------------------------------------------------------
# We split by position, NOT randomly.
#
# Why? Each training example is a sliding window of `block_size` tokens.
# If we split randomly, a window from the val set might overlap with a
# window in the train set, leaking information and making val loss
# artificially low. A positional cut avoids all overlap.

split_idx = int(len(token_ids) * TRAIN_SPLIT)

train_ids = token_ids[:split_idx]
val_ids   = token_ids[split_idx:]

print(f"Train tokens: {len(train_ids):,}  |  Val tokens: {len(val_ids):,}")

# ---------------------------------------------------------------------------
# 4. Write .bin files
# ---------------------------------------------------------------------------
# numpy uint16: covers vocab sizes up to 65,535 and uses half the space
# of int32. For BPE vocab_size=1000 this is more than sufficient.

def write_bin(ids: list, path: str) -> None:
    arr = np.array(ids, dtype=np.uint16)

    # numpy.memmap writes directly to disk in a flat binary format.
    # Shape (N,) means N consecutive uint16 values — no header, no padding.
    # A reader opens it with the same dtype and shape to get the array back.
    fp = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()   # make sure OS writes all pages to disk before we exit
    print(f"Written: {path}  ({len(arr):,} tokens, {arr.nbytes / 1024:.1f} KB)")

write_bin(train_ids, str(TRAIN_BIN))
write_bin(val_ids,   str(VAL_BIN))

print("\nDone. You can now run training with pre-tokenized data.")
print("Load in training with:")
print("    import numpy as np")
print("    train_ids = np.memmap('data/train.bin', dtype=np.uint16, mode='r')")
