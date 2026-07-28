import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

    def forward(self, token_ids):
        _, seq_len = token_ids.shape

        token_embeddings = self.token_embedding(token_ids)

        positions = torch.arange(seq_len, device=token_ids.device)
        positional_embeddings = self.position_embedding(positions)

        x = token_embeddings + positional_embeddings
        return x
