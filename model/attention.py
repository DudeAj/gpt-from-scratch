import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Applied to attention weights AFTER softmax.
        # This randomly drops some token-to-token connections,
        # forcing the model not to over-rely on specific positions.
        self.attn_dropout = nn.Dropout(dropout)

        # Applied to the output of this entire attention block.
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # (B, T, d_model) -> (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, H, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Causal mask: token i can only attend to positions 0..i
        # tril gives a lower-triangular matrix of ones; positions above
        # the diagonal are future tokens — we set them to -inf so
        # softmax makes their weight effectively 0.
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)  # <-- dropout on attention weights

        output = attention @ V  # (B, H, T, head_dim)

        # Merge heads back: (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.resid_dropout(self.w_o(output))  # <-- dropout on output
