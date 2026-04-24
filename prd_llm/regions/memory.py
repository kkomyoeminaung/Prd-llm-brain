"""
Memory Region - Associative Memory and Recall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCognitiveRegion


class MemoryRegion(BaseCognitiveRegion):
    """Associative memory region."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int, memory_size: int = 4096):
        super().__init__(region_id, d_model, d_ff, "Memory")
        
        self.memory_size = memory_size
        self.register_buffer('memory_keys', torch.randn(memory_size, d_model) / math.sqrt(d_model))
        self.register_buffer('memory_values', torch.randn(memory_size, d_model) / math.sqrt(d_model))
        self.register_buffer('memory_ages', torch.zeros(memory_size))
        
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self.read_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        
        similarities = torch.matmul(x, self.memory_keys.T) / math.sqrt(self.d_model)
        read_weights = F.softmax(similarities / 0.1, dim=-1)
        memory_out = torch.matmul(read_weights, self.memory_values)
        
        write_prob = self.write_gate(x)
        write_mask = (write_prob > 0.5).float()
        
        if write_mask.sum() > 0:
            oldest_idx = self.memory_ages.argsort()[:int(write_mask.sum().item())]
            for idx in oldest_idx:
                self.memory_keys[idx] = x.mean(dim=[0, 1])
                self.memory_values[idx] = memory_out.mean(dim=[0, 1])
                self.memory_ages[idx] = 0
        
        self.memory_ages += 1
        output = base_out + self.read_gate(x) * memory_out
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
