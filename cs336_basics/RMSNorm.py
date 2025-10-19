import torch
from torch import nn
from einops import rearrange, einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        result = self.weight * (x / self.RMS(x))
        # Return the result in the original dtype
        return result.to(in_dtype)



    def RMS(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
