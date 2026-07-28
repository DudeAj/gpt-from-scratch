from turtle import forward

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, expension=4):
        super().__init__()
        hidden_dim = d_model * expension

        #(B,T,768) - > (B,T,3072)
        self.fc1 = nn.Linear(
            d_model, 
            hidden_dim
        )

        self.activation = nn.GELU()

        #(B,T,3072) - > (B,T,768)
        self.fc2 = nn.Linear(
            hidden_dim,
            d_model
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

