"""
Language Region - Natural Language Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCognitiveRegion


class LanguageRegion(BaseCognitiveRegion):
    """Language processing region with grammar awareness."""
    
    def __init__(self, region_id: int, d_model: int, d_ff: int):
        super().__init__(region_id, d_model, d_ff, "Language")
        
        self.grammar_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.grammar_gate = nn.Linear(d_model, 1)
        self.lang_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, x: torch.Tensor, update_plasticity: bool = True) -> torch.Tensor:
        if not self.is_active:
            return torch.zeros_like(x)
        
        base_out = super().forward(x, update_plasticity=False)
        grammar_features = self.grammar_proj(x)
        grammar_weight = torch.sigmoid(self.grammar_gate(x))
        output = base_out + grammar_features * grammar_weight + self.lang_embedding
        
        if update_plasticity:
            output = self.plasticity(output, update=True)
        
        return output
