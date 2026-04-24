"""
Code Region - Programming and Code Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCognitiveRegion


class CodeRegion(BaseCognitiveRegion):
    """Code generation region with syntax awareness."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Code")
        
        self.syntax_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.api_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        syntax_weight = self.syntax_detector(x)
        api_out = self.api_proj(x)
        output = base_out * syntax_weight + 0.2 * api_out
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
