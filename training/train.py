import os

from torch.utils.data import DataLoader

from tokenizer.bpe_tokenizer import BPETokenizer
from dataset.text_dataset import GPTDataset

from configs.config import (
    VOCAB_SIZE,
    BLOCK_SIZE,
    BATCH_SIZE
)

TOKENIZER_PATH = "checkpoints/tokenizer.json"

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

tokenizer = BPETokenizer()

if os.path.exists(TOKENIZER_PATH):

    tokenizer.load(TOKENIZER_PATH)

else:

    tokenizer.train(text, VOCAB_SIZE)

    tokenizer.save(TOKENIZER_PATH)

tokens = tokenizer.encode(text)

dataset = GPTDataset(
    tokens=tokens,
    block_size=BLOCK_SIZE
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for x, y in train_loader:

    print("Input Shape :", x.shape)
    print("Target Shape:", y.shape)

    break