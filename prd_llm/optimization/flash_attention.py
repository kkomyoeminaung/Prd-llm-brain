"""
Flash Attention - Optimized attention computation
2-4x speed improvement, O(n) memory instead of O(n²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlashAttention(nn.Module):
    """
    Flash Attention implementation using PyTorch scaled_dot_product_attention
    """
    
    def __init__(self, use_flash: bool = True):
        super().__init__()
        self.use_flash = use_flash
        
    def forward(self, q, k, v, is_causal: bool = True):
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q, k, v, 
                is_causal=is_causal
            )
        else:
            # Fallback
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
            if is_causal:
                T = q.size(-2)
                mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v)
