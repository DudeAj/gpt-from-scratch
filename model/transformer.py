import torch
import torch.nn as nn
from model.layernorm import LayerNorm
from model.feedforward import FeedForward
from model.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.ln1 = LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ln2 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model)

    def forward(self,x):
        #input (B,T,d_model)

        norm_x = self.ln1(x)

        attention_output = self.attention(norm_x)

        x = x + attention_output

        norm_x = self.ln2(x)

        ffn_output = self.ffn(norm_x)

        x = x + ffn_output

        return x
