import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5 ):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - mean)/torch.sqrt(variance+self.eps)
        y = self.gamma * x_hat + self.beta

        return y



    
