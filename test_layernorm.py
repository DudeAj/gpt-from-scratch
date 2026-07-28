import torch
from model.layernorm import LayerNorm

layernorm = LayerNorm(32)

x = torch.randn(
    2,
    8,
    32
)

y = layernorm(x)

print(x.shape)
print(y.shape)