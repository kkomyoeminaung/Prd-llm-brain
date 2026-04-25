"""
Reasoning Region - Logical and Causal Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCognitiveRegion


class ReasoningRegion(BaseCognitiveRegion):
    """Logical reasoning region with causal inference."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Reasoning")
        
        self.causal_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.cot_memory = None
        self.cot_gate = nn.Linear(d_model, 1)
        self.patthana_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        causal_out, _ = self.causal_attention(x, x, x)
        
        if self.cot_memory is not None:
            cot_gate = torch.sigmoid(self.cot_gate(x))
            # Ensure cot_memory is [1, 1, D] for clean broadcasting
            mem = self.cot_memory
            if mem.dim() == 3:
                mem = mem.mean(dim=[0, 1], keepdim=True)
            elif mem.dim() == 2:
                mem = mem.mean(dim=0, keepdim=True).unsqueeze(0)
            
            base_out = base_out + cot_gate * 0.3 * mem
        
        patthana_weight = self.patthana_gate(x)
        output = (base_out + causal_out) * patthana_weight
        # Fix: Pool memory to [1, 1, D] to allow broadcasting across any Batch/Sequence size changes
        self.cot_memory = output.mean(dim=[0, 1], keepdim=True).detach()
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
