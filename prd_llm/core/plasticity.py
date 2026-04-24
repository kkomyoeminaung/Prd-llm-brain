"""
Synaptic Plasticity - Hebbian Learning Module
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import deque


class PlasticityAdapter(nn.Module):
    """
    Hebbian plasticity layer.
    Learning rule: ΔW = η × (pre × post)
    """
    
    def __init__(self, d_model: int, learning_rate: float = 0.01, decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.decay = decay
        
        self.plastic_weights = nn.Parameter(torch.zeros(d_model, d_model))
        self.plastic_gate = nn.Parameter(torch.zeros(d_model))
        
        self.register_buffer('pre_trace', torch.zeros(d_model))
        self.register_buffer('post_trace', torch.zeros(d_model))
        self.register_buffer('hebbian_accum', torch.zeros(d_model, d_model))
        
        self.update_count = 0
        self.plasticity_magnitudes = deque(maxlen=100)
        
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        B, T, D = x.shape
        
        if self.plastic_weights.abs().sum() > 0:
            adapted = torch.matmul(x, self.plastic_weights.tanh())
        else:
            adapted = x
        
        gate = torch.sigmoid(self.plastic_gate)
        output = adapted * (1 + gate.unsqueeze(0).unsqueeze(0))
        
        if update and self.training:
            self._hebbian_update(x, output)
        
        return output
    
    def _hebbian_update(self, pre: torch.Tensor, post: torch.Tensor):
        B, T, D = pre.shape
        
        pre_avg = pre.mean(dim=[0, 1])
        post_avg = post.mean(dim=[0, 1])
        
        self.pre_trace = self.decay * self.pre_trace + (1 - self.decay) * pre_avg
        self.post_trace = self.decay * self.post_trace + (1 - self.decay) * post_avg
        
        hebbian_delta = self.learning_rate * torch.outer(self.pre_trace, self.post_trace)
        self.hebbian_accum += hebbian_delta
        self.update_count += 1
        
        if self.update_count >= 10:
            self.plastic_weights.data += self.hebbian_accum
            self.plastic_weights.data = torch.clamp(self.plastic_weights.data, -1.0, 1.0)
            self.hebbian_accum.zero_()
            self.update_count = 0
            
            mag = hebbian_delta.abs().mean().item()
            self.plasticity_magnitudes.append(mag)
    
    def get_stats(self) -> Dict:
        return {
            'mean_weight_magnitude': self.plastic_weights.abs().mean().item(),
            'gate_value': torch.sigmoid(self.plastic_gate).mean().item(),
            'plasticity_activity': sum(self.plasticity_magnitudes) / len(self.plasticity_magnitudes) if self.plasticity_magnitudes else 0,
            'update_count': self.update_count,
        }
