import math
from turtle import forward

import torch 
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()


        assert d_model % num_heads == 0 # its will throw error if d_model is not totally dividable by num_heads 
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # key, query, value and output matrix
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, T, _ = x.shape
        #shape will be (B, T, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        #Reshape 
        # initial (B, T, 12, 64)
        # after transpose (B, 12, T, 64)
        Q = Q.view(B,T,self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B,T,self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B,T,self.num_heads, self.head_dim).transpose(1,2)

        #after multiplication (B, H, T, D) x (B, H, D, T) = (B, H, T, T)
        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T,T, device=x.device))

        scores = scores.masked_fill(mask==0, float('-inf'))

        attention = F.softmax(scores,dim=-1)

        output = attention @ V

        output = output.transpose(1,2).contiguous()

        output = output.view(
            B,
            T,
            self.d_model
        )

        return self.w_o(output)







        

