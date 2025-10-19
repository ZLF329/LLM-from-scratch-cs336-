import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.Linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = y * torch.sigmoid(y)
        z = self.w3(x)
        x = y * z
        x = self.w2(x)
        return x