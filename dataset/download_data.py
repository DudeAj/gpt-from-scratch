"""
download_data.py
----------------
Downloads WikiText-103 and DailyDialog from HuggingFace Hub and saves
them as plain text files under data/wikitext/ and data/dailydialog/.

This is a one-time setup step — run before pretrain.py or finetune.py.

Usage:
    python -m dataset.download_data              # both datasets
    python -m dataset.download_data --wiki       # WikiText-103 only
    python -m dataset.download_data --dialog     # DailyDialog only
"""

import argparse
import os
import json
from datasets import load_dataset

from configs.config import WIKITEXT_DIR, DAILYDIALOG_DIR, HUMAN_TOKEN, ASSISTANT_TOKEN, END_TOKEN


# ---------------------------------------------------------------------------
# WikiText-103
# ---------------------------------------------------------------------------

def download_wikitext():
    """
    WikiText-103 is a collection of ~28k Wikipedia articles.
    Total size: ~500MB of clean, tokenised prose.

    We save three splits (train / validation / test) as plain .txt files,
    one article per line, so our WikiTextDataset can stream them cheaply.

    Why plain text and not HF Arrow format?
    Arrow caches are huge. Plain text is simpler to inspect and load,
    and our custom BPE tokenizer can't consume Arrow natively anyway.
    """
    print("Downloading WikiText-103 …")
    os.makedirs(str(WIKITEXT_DIR), exist_ok=True)

    # 'wikitext-103-v1' is the standard benchmark split
    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)

    for split in ("train", "validation", "test"):
        out_path = str(WIKITEXT_DIR / f"{split}.txt")
        if os.path.exists(out_path):
            print(f"  {split}.txt already exists — skipping")
            continue

        print(f"  Writing {split}.txt …", end=" ", flush=True)
        lines_written = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for row in dataset[split]:
                text = row["text"].strip()

                # Skip empty lines and section headers ( = Header = )
                # We keep article text only — headers add noise without
                # useful language signal for a small vocabulary.
                if not text or text.startswith(" = "):
                    continue

                f.write(text + "\n")
                lines_written += 1

        print(f"{lines_written:,} lines")

    print("  WikiText-103 done.\n")


# ---------------------------------------------------------------------------
# DailyDialog
# ---------------------------------------------------------------------------

def download_dailydialog():
    """
    DailyDialog contains ~13k everyday English conversations,
    each with 2–10 turns. It's clean, manually annotated, and
    small enough for fine-tuning a small GPT.

    Format we write to disk:
        Each conversation becomes one JSON line (JSONL):
        {"turns": ["Hello!", "Hi there, how are you?", "I'm good, thanks!"]}

    Why JSONL?
        Easy to stream line-by-line without loading the entire file,
        and preserves the turn structure we need for formatting.

    During fine-tuning we'll format alternating turns as:
        <human> Hello! </s> <assistant> Hi there, how are you? </s>
    so the model learns the conversation pattern.
    """
    print("Downloading DailyDialog …")
    os.makedirs(str(DAILYDIALOG_DIR), exist_ok=True)

    dataset = load_dataset("daily_dialog", trust_remote_code=True)

    for split in ("train", "validation", "test"):
        out_path = str(DAILYDIALOG_DIR / f"{split}.jsonl")
        if os.path.exists(out_path):
            print(f"  {split}.jsonl already exists — skipping")
            continue

        print(f"  Writing {split}.jsonl …", end=" ", flush=True)
        count = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for row in dataset[split]:
                turns = [t.strip() for t in row["dialog"] if t.strip()]
                if len(turns) < 2:
                    continue   # skip single-turn entries

                line = json.dumps({"turns": turns}, ensure_ascii=False)
                f.write(line + "\n")
                count += 1

        print(f"{count:,} conversations")

    print("  DailyDialog done.\n")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_stats():
    print("=" * 50)
    print("Dataset summary")
    print("=" * 50)

    for path in (
        WIKITEXT_DIR / "train.txt",
        WIKITEXT_DIR / "validation.txt",
        DAILYDIALOG_DIR / "train.jsonl",
        DAILYDIALOG_DIR / "validation.jsonl",
    ):
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            with open(str(path)) as f:
                lines = sum(1 for _ in f)
            print(f"  {path.name:<25} {lines:>8,} lines  {size_mb:6.1f} MB")
        else:
            print(f"  {path.name:<25} not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki",   action="store_true", help="Download WikiText-103 only")
    parser.add_argument("--dialog", action="store_true", help="Download DailyDialog only")
    args = parser.parse_args()

    # Default: download both
    both = not args.wiki and not args.dialog

    if args.wiki or both:
        download_wikitext()

    if args.dialog or both:
        download_dailydialog()

    print_stats()
