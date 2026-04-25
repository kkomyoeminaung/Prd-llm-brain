"""
Mathematics Region - Numerical and Symbolic Computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCognitiveRegion


class MathRegion(BaseCognitiveRegion):
    """Mathematical computation region."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Mathematics")
        
        self.numerical_encoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.GELU(),
        )
        self.add_gate = nn.Linear(d_model, d_model)
        self.mul_gate = nn.Linear(d_model, d_model)
        self.precision_gate = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        numerical = self.numerical_encoder(x)
        add_op = self.add_gate(x)
        mul_op = self.mul_gate(x) * x
        precision = torch.sigmoid(self.precision_gate(x))
        
        output = base_out + numerical + 0.5 * add_op + 0.3 * mul_op
        output = output * precision
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
