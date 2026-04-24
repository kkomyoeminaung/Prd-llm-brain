"""
Vision Region - Spatial and Pattern Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCognitiveRegion


class VisionRegion(BaseCognitiveRegion):
    """Visual/spatial processing region."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int, max_len: int = 512):
        super().__init__(region_id, d_model, d_ff, "Vision")
        
        self.register_buffer('spatial_coords', self._create_spatial_grid(max_len, d_model))
        self.pattern_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
    def _create_spatial_grid(self, max_len: int, d_model: int) -> torch.Tensor:
        grid = torch.zeros(max_len, d_model)
        for i in range(max_len):
            for j in range(0, d_model, 2):
                grid[i, j] = math.sin(i / (10000 ** (j / d_model)))
                if j + 1 < d_model:
                    grid[i, j + 1] = math.cos(i / (10000 ** (j / d_model)))
        return grid.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        B, T, D = x.shape
        x_with_spatial = x + self.spatial_coords[:, :T, :].to(x.device)
        output = self.pattern_net(x_with_spatial)
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
