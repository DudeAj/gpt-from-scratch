import torch
from model.feedforward import FeedForward


ffn = FeedForward(d_model=32)

x = torch.randn(
    2,
    8,
    32
)

y = ffn(x)

print(x.shape)
print(y.shape)