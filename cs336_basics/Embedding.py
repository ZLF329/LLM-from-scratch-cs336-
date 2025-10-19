import torch
from torch import nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
         super().__init__()
         self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(num_embeddings,embedding_dim), mean=0.0, std=1.0, a=-3.0, b=3.0))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
