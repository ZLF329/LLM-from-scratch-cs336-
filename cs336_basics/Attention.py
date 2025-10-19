import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding as RoPE
from cs336_basics.Linear import Linear

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float=None, max_seq_len: int=None, use_rope: bool = False, token_positions: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.o_linear = Linear(d_model, d_model)
        self.token_positions = token_positions
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len) if use_rope else None
    

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q,k,v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        q = rearrange(q, 'b s (h d_k) -> (b h) s d_k', h=self.num_heads)
        k = rearrange(k, 'b s (h d_k) -> (b h) s d_k', h=self.num_heads)
        v = rearrange(v, 'b s (h d_k) -> (b h) s d_k', h=self.num_heads)
        if not self.token_positions:
            self.token_positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
            self.token_positions = self.token_positions.repeat_interleave(self.num_heads, dim=0)

        q = self.rope(q, self.token_positions) if self.rope else q
        k = self.rope(k, self.token_positions) if self.rope else k
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = rearrange(attn_output, '(b h) s d_k -> b s (h d_k)', b=x.size(0), h=self.num_heads)
        output = self.o_linear(attn_output)
        return output



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # q, K:(batch_size, ..., seq_len, d_k), `v`: (batch_size, ..., seq_len, d_v)
    score = einsum(q, k, '... q d_k, ... k d_k -> ... q k')  # (..., query_len, key_len)
    score = score.masked_fill(~mask, float('-inf'))
    softmax_scores = softmax(score / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32)), dim=-1)  # (..., query_len, key_len)
    full_product = einsum(softmax_scores, v, '... q k, ... k d_v -> ... q d_v')  # (..., query_len, d_v)
    return full_product

