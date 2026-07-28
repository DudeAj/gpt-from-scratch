"""
dialog_dataset.py
-----------------
PyTorch Dataset for fine-tuning on DailyDialog conversations.

Conversation format
-------------------
Each training example is a complete dialogue formatted as a single
token sequence using special sentinel tokens:

    <human> Hello! </s> <assistant> Hi there! </s> <human> How are you? </s> <assistant> I'm great, thanks! </s>

The model learns to predict every next token, but we only compute loss
on the ASSISTANT turns — we don't penalise the model for what the human
says, only for what the assistant replies.  This is the same approach
used in InstructGPT and LLaMA fine-tuning.

Why mask human turns?
    If we compute loss on all tokens, the model learns to "predict" human
    questions just as hard as assistant replies.  That wastes capacity and
    makes the model no better at generating responses specifically.
    Masking human turns focuses all gradient signal on reply quality.

Special tokens
--------------
We add <human>, <assistant>, </s> to GPT-2's vocabulary so the model
can learn the conversation structure boundary.  These are new token IDs
appended after the existing 50,257.
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.processors import TemplateProcessing

from configs.config import (
    DAILYDIALOG_DIR,
    HUMAN_TOKEN, ASSISTANT_TOKEN, END_TOKEN,
)


# ---------------------------------------------------------------------------
# Tokenizer with special tokens
# ---------------------------------------------------------------------------

def load_dialog_tokenizer() -> tuple[HFTokenizer, int]:
    """
    Load GPT-2 tokenizer and add three special tokens.

    Returns
    -------
    tokenizer  : extended HFTokenizer
    vocab_size : new vocab size (50,257 + 3 special tokens)
    """
    tokenizer = HFTokenizer.from_pretrained("gpt2")

    # Add special tokens — HF tokenizers returns the number added
    added = tokenizer.add_special_tokens([
        HUMAN_TOKEN,
        ASSISTANT_TOKEN,
        END_TOKEN,
    ])
    print(f"  Added {added} special tokens → vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer, tokenizer.get_vocab_size()


# ---------------------------------------------------------------------------
# Conversation formatter
# ---------------------------------------------------------------------------

def format_conversation(turns: list[str], tokenizer: HFTokenizer) -> tuple[list[int], list[int]]:
    """
    Encode a list of alternating [human, assistant, human, ...] turns
    into a flat token sequence with a parallel loss-mask.

    Parameters
    ----------
    turns     : ["Hello!", "Hi there!", "How are you?", ...]
    tokenizer : extended GPT-2 tokenizer

    Returns
    -------
    token_ids : flat list of int token IDs
    loss_mask : 1 where loss should be computed (assistant turns), 0 elsewhere

    Example
    -------
    turns = ["Hi", "Hello there", "Bye"]

    token_ids → [H_tok, hi_ids..., end_tok,  A_tok, hello_ids..., end_tok,
                 H_tok, bye_ids..., end_tok]
    loss_mask → [0,     0,...,      0,        1,     1,...,        1,
                 0,     0,...,      0]
    """
    token_ids = []
    loss_mask = []

    human_id     = tokenizer.token_to_id(HUMAN_TOKEN)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    end_id       = tokenizer.token_to_id(END_TOKEN)

    for i, turn in enumerate(turns):
        is_assistant = (i % 2 == 1)   # odd indices = assistant

        # Prefix token: <human> or <assistant>
        prefix_id  = assistant_id if is_assistant else human_id
        turn_ids   = tokenizer.encode(turn).ids

        segment    = [prefix_id] + turn_ids + [end_id]
        token_ids.extend(segment)

        # Compute loss only on assistant turns (including the </s> closer)
        mask_val   = 1 if is_assistant else 0
        loss_mask.extend([mask_val] * len(segment))

    return token_ids, loss_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DialogDataset(Dataset):
    """
    Fine-tuning dataset built from DailyDialog JSONL files.

    Each item is a tuple (x, y, mask) where:
      x    : input token IDs   (block_size,)
      y    : target token IDs  (block_size,)  — x shifted right by 1
      mask : float loss mask   (block_size,)  — 1.0 on assistant positions

    Long conversations are truncated to block_size + 1 tokens.
    Short conversations are padded to block_size + 1 with the GPT-2 EOS
    token (id=50256) and masked out (mask=0).

    Parameters
    ----------
    split      : "train" | "validation" | "test"
    block_size : context window length
    tokenizer  : extended GPT-2 tokenizer (from load_dialog_tokenizer)
    """

    def __init__(self, split: str, block_size: int, tokenizer: HFTokenizer):
        self.block_size  = block_size
        self.tokenizer   = tokenizer
        self.examples    = []   # list of (token_ids, loss_mask)

        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            eos_id = 0   # fallback

        jsonl_path = str(DAILYDIALOG_DIR / f"{split}.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj   = json.loads(line)
                turns = obj["turns"]
                ids, mask = format_conversation(turns, tokenizer)

                # Truncate to block_size + 1 (need +1 to produce x and y)
                ids  = ids [:block_size + 1]
                mask = mask[:block_size + 1]

                # Pad short sequences with EOS (masked out)
                pad_len = (block_size + 1) - len(ids)
                if pad_len > 0:
                    ids  = ids  + [eos_id] * pad_len
                    mask = mask + [0]      * pad_len

                self.examples.append((ids, mask))

        print(f"  {split}: {len(self.examples):,} conversations loaded")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, mask = self.examples[idx]

        tokens   = torch.tensor(ids,  dtype=torch.long)
        mask_t   = torch.tensor(mask, dtype=torch.float)

        x = tokens[:-1]    # input:  positions 0..block_size-1
        y = tokens[1:]     # target: positions 1..block_size
        m = mask_t[1:]     # mask aligned to target positions

        return x, y, m


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def _ensure_downloaded():
    """Download DailyDialog if the JSONL files aren't present yet."""
    train_path = DAILYDIALOG_DIR / "train.jsonl"
    if not train_path.exists():
        print("  DailyDialog not found — downloading now …")
        from dataset.download_data import download_dailydialog
        download_dailydialog()


def make_dialog_datasets(block_size: int):
    """
    Returns (train_dataset, val_dataset, vocab_size).
    Call this from finetune.py.
    """
    _ensure_downloaded()

    print("Loading dialog tokenizer …")
    tokenizer, vocab_size = load_dialog_tokenizer()

    print("Building DailyDialog datasets …")
    train_ds = DialogDataset("train",      block_size, tokenizer)
    val_ds   = DialogDataset("validation", block_size, tokenizer)

    return train_ds, val_ds, vocab_size, tokenizer
