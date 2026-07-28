import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion

        # Two-layer MLP: expand to hidden_dim, then project back.
        # The expansion (typically 4x) lets the model learn richer
        # per-token transformations than a single linear layer could.
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

        # Dropout after the second projection, before the residual add.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)  # <-- dropout on FFN output
