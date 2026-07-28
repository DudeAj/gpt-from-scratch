import torch.nn as nn

from model.layernorm import LayerNorm
from model.feedforward import FeedForward
from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    One decoder-only transformer block:

        x -> LayerNorm -> MultiHeadAttention -> + (residual)
          -> LayerNorm -> FeedForward         -> + (residual)

    Pre-norm layout (LayerNorm before each sub-layer) is more
    stable to train than the original post-norm paper design.
    The residual connections let gradients flow directly back to
    early layers, solving the vanishing-gradient problem.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        # Pre-norm attention sub-layer
        x = x + self.attention(self.ln1(x))

        # Pre-norm feed-forward sub-layer
        x = x + self.ffn(self.ln2(x))

        return x
