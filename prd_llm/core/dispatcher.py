"""
Sparse Dispatcher - Routes inputs to active cognitive regions
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import deque


class SparseDispatcher(nn.Module):
    """Routes input tokens to selected cognitive regions."""
    
    def __init__(self, d_model: int, num_regions: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.num_regions = num_regions
        self.top_k = top_k
        
        self.combiner = nn.Linear(d_model * num_regions, d_model)
        self.combiner_norm = nn.LayerNorm(d_model)
        
        self.total_dispatches = 0
        self.active_region_counts = deque(maxlen=100)
        
    def forward(self, hidden: torch.Tensor, weights: torch.Tensor, 
                indices: torch.Tensor, regions: nn.ModuleList,
                update_plasticity: bool = True) -> Tuple[torch.Tensor, Dict]:
        
        B, T, D = hidden.shape
        
        # Activate selected regions
        active_ids = torch.unique(indices).tolist()
        for i, region in enumerate(regions):
            if i in active_ids:
                region.activate()
            else:
                region.deactivate()
        
        # Collect outputs
        region_outputs = []
        for i, region in enumerate(regions):
            out = region(hidden, update_plasticity=update_plasticity)
            region_outputs.append(out)
        
        # Apply routing weights
        weighted_output = torch.zeros_like(hidden)
        for b in range(B):
            for t in range(T):
                for k in range(self.top_k):
                    region_id = indices[b, t, k].item()
                    weight = weights[b, t, k].item()
                    if region_id < len(region_outputs):
                        weighted_output[b, t] += weight * region_outputs[region_id][b, t]
        
        # Combine all region outputs
        combined = torch.cat(region_outputs, dim=-1)
        context = self.combiner_norm(self.combiner(combined))
        output = weighted_output + 0.2 * context
        
        self.total_dispatches += 1
        self.active_region_counts.append(len(active_ids))
        
        stats = {
            'active_regions': active_ids,
            'num_active': len(active_ids),
            'active_percentage': (len(active_ids) / len(regions)) * 100,
            'avg_active_last_100': sum(self.active_region_counts) / len(self.active_region_counts) if self.active_region_counts else 0,
        }
        
        return output, stats
