import torch
import torch.nn as nn

from model.transformer import TransformerBlock
from model.embedding import GPTEmbedding
from model.layernorm import LayerNorm


class GPT(nn.Module):
    """
    Decoder-only GPT:

      token_ids
        -> token + positional embeddings   (GPTEmbedding)
        -> N x TransformerBlock            (attention + FFN)
        -> final LayerNorm
        -> linear projection to vocab      (lm_head)
        -> logits  shape: (B, T, vocab_size)

    Weight tying: lm_head shares weights with the token embedding
    matrix. This halves the parameters for that layer and has been
    shown empirically to improve perplexity.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = GPTEmbedding(vocab_size, max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, token_ids):
        x = self.embedding(token_ids)           # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                        # final layer norm
        logits = self.lm_head(x)                # (B, T, vocab_size)
        return logits
