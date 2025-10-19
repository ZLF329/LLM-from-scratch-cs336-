import torch
import torch.nn as nn
from einops import rearrange, einsum


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        pos = torch.arange(0, max_seq_len, device=device)
        angles = einsum(pos, freqs, 'n, d -> n d') # (max_seq_len, d_k/2)
        self.register_buffer('cos', torch.cos(angles), persistent=False)  
        self.register_buffer('sin', torch.sin(angles), persistent=False)  

    