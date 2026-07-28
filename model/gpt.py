import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.transformer import TransformerBlock
from model.embedding import GPTEmbedding
from model.layernorm import LayerNorm

class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            max_seq_len, 
            d_model, 
            num_heads, 
            num_layers
        ):
        super().__init__()

        self.embedding = GPTEmbedding(vocab_size, max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads) for _ in range(num_layers)
            ]
        ) 

        self.ln_f = LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = (self.embedding.token_embedding.weight)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits


       