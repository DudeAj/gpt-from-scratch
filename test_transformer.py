import torch

from model.transformer import TransformerBlock


block = TransformerBlock(
    d_model=32,
    num_heads=4,
)

x = torch.randn(
    2,
    8,
    32,
)

y = block(x)

print(x.shape)
print(y.shape)