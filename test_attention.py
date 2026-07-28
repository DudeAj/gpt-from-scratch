import torch
from model.attention import MultiHeadAttention

attention = MultiHeadAttention(
    d_model=32,
    num_heads=4
)

x = torch.randn(2,8,32)

y = attention(x)


print(x.shape)
print(y.shape)