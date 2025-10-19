import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.Rope import Rope


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.rope = Rope(theta, d_k, max_seq_len, device=device) # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.rope.cos[token_positions]  # (batch_size, seq_len, d_k/2)
        sin = self.rope.sin[token_positions]  # (batch_size, seq_len, d_k/2)
        x_even = x[..., 0::2]  # (batch, seq_len, dim/2)
        x_odd  = x[..., 1::2]  # (batch, seq_len, dim/2)
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated  = x_even * sin + x_odd * cos
        x_rotated = torch.stack((x_even_rotated, x_odd_rotated), dim=-1).flatten(-2)
        return x_rotated

        