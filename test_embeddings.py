import torch
from model.embedding import GPTEmbedding

embedding = GPTEmbedding(
    vocab_size=100,
    max_seq_len=16,
    d_model=32
)

token_ids = torch.randint(
    0,
    100,
    (2, 8),
)

output = embedding(token_ids)

print(token_ids.shape)
print(output.shape)