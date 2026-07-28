import torch

from model.gpt import GPT

model = GPT(
    vocab_size=100,
    max_seq_len=16,
    d_model=32,
    num_heads=4,
    num_layers=2,
)

tokens = torch.randint(
    0,
    100,
    (2,8),
)

logits = model(tokens)

print(tokens.shape)
print(logits.shape)