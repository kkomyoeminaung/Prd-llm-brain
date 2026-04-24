"""
Transformer Components - RMSNorm, RoPE, Attention, FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self.rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * config.n_heads == config.d_model
        
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, T)
        k = self.rope(k, T)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(causal_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_out = torch.matmul(attn, v)
        
        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, config):
        super().__init__()
        hidden = config.d_ff
        self.gate = nn.Linear(config.d_model, hidden, bias=False)
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.down(self.dropout(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Block"""
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
