"""
KV Cache Optimization - Reuse past computations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KVCache:
    """
    Key-Value cache for autoregressive generation
    """
    
    def __init__(self, num_layers: int = 8):
        self.num_layers = num_layers
        self.k_cache = [None] * num_layers
        self.v_cache = [None] * num_layers
        self.current_length = 0
        
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
        
        self.current_length = self.k_cache[layer_idx].shape[2]
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def reset(self):
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.current_length = 0
