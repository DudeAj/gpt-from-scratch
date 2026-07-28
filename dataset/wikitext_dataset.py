"""
wikitext_dataset.py
-------------------
PyTorch Dataset for WikiText-103 pretraining.

Design decisions
----------------
1. We use GPT-2's byte-level BPE tokenizer (vocab 50,257) instead of our
   hand-rolled BPE.  WikiText-103 has ~100M tokens; a 1000-token vocab
   would need ~4 sub-tokens per word and produce incoherent text at any
   model size.  GPT-2's tokenizer covers the full Unicode byte range so
   it never hits an unknown token.

2. The file is tokenized once on first use and cached as a numpy uint16
   array (same .bin format as prepare_data.py).  Subsequent runs load
   the cache instantly.

3. Sliding window: identical to GPTDataset — windows of `block_size`
   tokens, each shifted by 1 to form the (input, target) pair.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer as HFTokenizer

from configs.config import WIKITEXT_DIR, CHECKPOINT_DIR


# ---------------------------------------------------------------------------
# GPT-2 tokenizer loader
# ---------------------------------------------------------------------------

def load_gpt2_tokenizer() -> HFTokenizer:
    """
    Load the GPT-2 BPE tokenizer from HuggingFace Hub (cached locally).

    Why GPT-2's tokenizer?
      - Byte-level BPE: every possible byte is a valid token, so the
        tokenizer never produces UNK — even on code, maths, or emoji.
      - 50,257 tokens: fine-grained enough for a 768-dim model to build
        rich representations.
      - Pre-trained merge rules match WikiText-103 perfectly since GPT-2
        itself was trained on web text of the same style.
    """
    from tokenizers import Tokenizer as HFTokenizer

    # HuggingFace Hub caches the file at ~/.cache/huggingface/...
    # so this is only a network call on the very first run.
    tokenizer = HFTokenizer.from_pretrained("gpt2")
    return tokenizer


# ---------------------------------------------------------------------------
# Token cache helpers
# ---------------------------------------------------------------------------

def _cache_path(split: str) -> str:
    return str(WIKITEXT_DIR / f"{split}_tokens.bin")


def _tokenize_and_cache(split: str, tokenizer: HFTokenizer) -> np.ndarray:
    """
    Read the raw .txt, encode with GPT-2 tokenizer, write uint16 .bin.

    uint16 covers vocab sizes up to 65,535 — GPT-2's 50,257 fits fine.
    Saves ~50% memory vs int32.
    """
    txt_path   = str(WIKITEXT_DIR / f"{split}.txt")
    cache      = _cache_path(split)

    if os.path.exists(cache):
        print(f"  Loading cached {split} tokens: {cache}")
        return np.memmap(cache, dtype=np.uint16, mode="r")

    print(f"  Tokenizing {split}.txt …", end=" ", flush=True)
    all_ids = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # encode_batch would be faster but uses more RAM;
            # line-by-line is safer for large files
            ids = tokenizer.encode(line).ids
            all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    fp  = np.memmap(cache, dtype=np.uint16, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    print(f"{len(arr):,} tokens cached → {cache}")
    return np.memmap(cache, dtype=np.uint16, mode="r")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """
    Sliding-window dataset over a tokenized WikiText-103 split.

    Each item is a pair (x, y) where:
      x = tokens[i : i + block_size]      — input context
      y = tokens[i+1 : i + block_size+1]  — next-token targets

    The model is trained to predict every next token given all previous
    ones — this is the standard causal language modelling objective.

    Parameters
    ----------
    split      : "train" | "validation" | "test"
    block_size : context window length (must match model's max_seq_len)
    tokenizer  : GPT-2 HFTokenizer instance
    """

    def __init__(self, split: str, block_size: int, tokenizer: HFTokenizer):
        self.block_size = block_size
        self.tokens     = _tokenize_and_cache(split, tokenizer)

    def __len__(self) -> int:
        # Every position except the last `block_size` is a valid start
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int):
        # numpy memmap → Python list → tensor is the safest conversion
        # path when the array is memory-mapped (avoids copy issues)
        chunk = self.tokens[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # (block_size,)
        y = torch.from_numpy(chunk[1:])    # (block_size,)
        return x, y


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_wikitext_datasets(block_size: int):
    """
    Load tokenizer once and return (train_dataset, val_dataset).
    Call this from pretrain.py — don't instantiate WikiTextDataset directly.
    """
    print("Loading GPT-2 tokenizer …")
    tokenizer = load_gpt2_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size:,}")

    print("Building WikiText-103 datasets …")
    train_ds = WikiTextDataset("train",      block_size, tokenizer)
    val_ds   = WikiTextDataset("validation", block_size, tokenizer)

    return train_ds, val_ds, vocab_size
