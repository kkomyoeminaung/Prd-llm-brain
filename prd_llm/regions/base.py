"""
Base Cognitive Region - Specialized Expert Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..core.plasticity import PlasticityAdapter


class BaseCognitiveRegion(nn.Module):
    """Base class for all specialized cognitive regions."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int, name: str, 
                 plasticity_lr: float = 0.01):
        super().__init__()
        self.region_id = region_id
        self.d_model = d_model
        self.name = name
        self.is_active = False
        self.activation_count = 0
        self.total_compute_time = 0.0
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Linear(d_ff // 2, d_model),
        )
        
        self.gate = nn.Linear(d_model, 1)
        self.plasticity = PlasticityAdapter(d_model, learning_rate=plasticity_lr)
        self.register_buffer('short_term_memory', torch.zeros(1, 1, d_model))
        self.memory_decay = 0.9
        
    def activate(self):
        self.is_active = True
        
    def deactivate(self):
        self.is_active = False
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        import time
        start = time.time()
        self.activation_count += 1
        
        base_out = self.net(x)
        gate_value = torch.sigmoid(self.gate(x))
        gated_out = base_out * gate_value
        plastic_out = self.plasticity(gated_out, update=update_plasticity)
        
        memory = self.short_term_memory.expand_as(plastic_out) * self.memory_decay
        output = plastic_out + 0.1 * memory
        self.short_term_memory = output.mean(dim=[0, 1], keepdim=True).detach()
        
        self.total_compute_time += time.time() - start
        return output
    
    def get_stats(self) -> Dict:
        return {
            'region_id': self.region_id,
            'name': self.name,
            'is_active': self.is_active,
            'activation_count': self.activation_count,
            'compute_time': self.total_compute_time,
            'plasticity': self.plasticity.get_stats(),
        }
