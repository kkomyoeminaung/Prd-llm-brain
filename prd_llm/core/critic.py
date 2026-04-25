"""
Critic Layer - Self-Correction and Hallucination Prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from collections import deque


class CriticLayer(nn.Module):
    """Self-correction mechanism with confidence scoring."""
    
    def __init__(self, d_model: int, threshold: float = 0.5, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold
        
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.consistency_proj = nn.Linear(d_model * 2, d_model)
        self.correction_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1)
        )
        
        self.confidence_history = deque(maxlen=1000)
        self.correction_history = deque(maxlen=1000)
        self.total_corrections = 0
        
    def forward(self, x: torch.Tensor, previous_output: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        B, T, D = x.shape
        
        confidence = self.confidence_net(x)
        self.confidence_history.append(confidence.mean().item())
        
        consistency_score = torch.ones(B, T, 1, device=x.device)
        if previous_output is not None:
            # Match sequence lengths if they differ
            T_x = x.size(1)
            T_p = previous_output.size(1)
            
            if T_x != T_p:
                if T_x < T_p:
                    p_sliced = previous_output[:, -T_x:, :]
                    x_sliced = x
                else:
                    p_sliced = previous_output
                    x_sliced = x[:, -T_p:, :]
                combined = torch.cat([x_sliced, p_sliced], dim=-1)
            else:
                combined = torch.cat([x, previous_output], dim=-1)
                
            consistency_feat = self.consistency_proj(combined)
            consistency_score = torch.sigmoid(consistency_feat.mean(dim=-1, keepdim=True))
        
        reliability = confidence * consistency_score
        needs_correction = (reliability < self.threshold).float()
        
        correction = self.correction_net(x)
        corrected_output = x + needs_correction * correction * 0.3
        
        correction_rate = needs_correction.float().mean().item()
        self.correction_history.append(correction_rate)
        if needs_correction.sum() > 0:
            self.total_corrections += 1
        
        stats = {
            'confidence_mean': confidence.mean().item(),
            'consistency_mean': consistency_score.mean().item(),
            'reliability_mean': reliability.mean().item(),
            'needs_correction_pct': correction_rate * 100,
            'total_corrections': self.total_corrections,
            'avg_confidence': sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.5,
        }
        
        return corrected_output, confidence, stats
    
    def is_confident(self, confidence: torch.Tensor) -> bool:
        return confidence.mean().item() >= self.threshold
    
    def reset(self):
        self.confidence_history.clear()
        self.correction_history.clear()
        self.total_corrections = 0
