import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):

        x = torch.tensor(
            self.tokens[idx: idx + self.block_size],
            dtype=torch.long
        )

        y = torch.tensor(
            self.tokens[idx + 1: idx + self.block_size + 1],
            dtype=torch.long
        )

        return x, y