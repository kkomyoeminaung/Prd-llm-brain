"""
Motor Region - Action and Response Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .base import BaseCognitiveRegion


class MotorRegion(BaseCognitiveRegion):
    """Motor control region for sequence smoothing."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Motor")
        
        self.action_planner = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.response_buffer = deque(maxlen=5)
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        action = self.action_planner(x)
        # Fix: Store pooled action to allow broadcasting across different sequence lengths in generation
        self.response_buffer.append(action.mean(dim=[0, 1], keepdim=True).detach())
        
        if len(self.response_buffer) > 1:
            prev_avg = sum(self.response_buffer) / len(self.response_buffer)
            # prev_avg is now [1, 1, D], action is [B, T, D]. They are broadcast-compatible.
            action = 0.7 * action + 0.3 * prev_avg
        
        output = base_out + action
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
