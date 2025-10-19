import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.PositionwiseFeedForward import PositionwiseFeedForward
from cs336_basics.Attention import MultiheadSelfAttention
from cs336_basics.Attention import softmax
from cs336_basics.Linear import Linear
class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x + self.attn(self.rms_norm1(x))
        x = x + self.ff(self.rms_norm2(x))
        return x
        
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,d_model: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model)
        self.output_embeddings = Linear(d_model, vocab_size)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x) 
        x = self.output_embeddings(self.rms_norm(x))
        return x
        