"""
Global Brain Router - Sparse Activation Gatekeeper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict
from collections import deque


class GlobalBrainRouter(nn.Module):
    """
    Central gatekeeper that activates only top-k regions.
    """
    
    def __init__(self, d_model: int, num_regions: int, top_k: int = 2, temperature: float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.num_regions = num_regions
        self.top_k = top_k
        self.temperature = temperature
        
        self.gate = nn.Linear(d_model, num_regions, bias=False)
        self.region_embeddings = nn.Parameter(
            torch.randn(num_regions, d_model) / math.sqrt(d_model)
        )
        
        self.usage_counter = torch.zeros(num_regions)
        self.routing_history = deque(maxlen=1000)
        self.total_routings = 0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        content_scores = self.gate(x)
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_regions = F.normalize(self.region_embeddings, p=2, dim=-1)
        semantic_scores = torch.matmul(norm_x, norm_regions.T)
        
        combined_scores = content_scores + 0.3 * semantic_scores
        combined_scores = combined_scores / self.temperature
        
        weights, indices = torch.topk(combined_scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        unique_indices = torch.unique(indices)
        for idx in unique_indices:
            self.usage_counter[idx] += 1
        
        self.total_routings += 1
        return weights, indices, combined_scores
    
    def get_load_balance(self) -> float:
        if self.usage_counter.sum() == 0:
            return 0.0
        usage = self.usage_counter / self.usage_counter.sum()
        cv = usage.std() / (usage.mean() + 1e-8)
        return cv.item()
    
    def get_stats(self) -> Dict:
        return {
            'total_routings': self.total_routings,
            'top_k': self.top_k,
            'num_regions': self.num_regions,
            'load_balance': self.get_load_balance(),
        }
