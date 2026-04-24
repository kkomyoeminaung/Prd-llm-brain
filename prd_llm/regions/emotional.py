"""
Emotional Region - Value and Reward Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .base import BaseCognitiveRegion


class EmotionalRegion(BaseCognitiveRegion):
    """Emotional/value processing region."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Emotional")
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Linear(d_ff // 2, 1),
            nn.Sigmoid()
        )
        self.emotion_memory = deque(maxlen=50)
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        reward = self.reward_predictor(x)
        
        self.emotion_memory.append(reward.mean().item())
        mood = sum(self.emotion_memory) / len(self.emotion_memory)
        
        output = base_out * (1 + 0.3 * mood)
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
