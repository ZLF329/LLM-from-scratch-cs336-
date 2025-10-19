from torch import nn
import torch
from einops import rearrange, einsum
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(out_features, in_features), mean=0.0, std=std, a=-3.0, b=3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... i, o i -> ... o')
