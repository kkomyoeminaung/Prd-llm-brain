"""
PRD-LLM Complete Brain Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .config import PRDLLMConfig
from .core.transformer import TransformerBlock, RMSNorm
from .core.router import GlobalBrainRouter
from .core.dispatcher import SparseDispatcher
from .core.critic import CriticLayer
from .regions.reasoning import ReasoningRegion
from .regions.language import LanguageRegion
from .regions.math import MathRegion
from .regions.memory import MemoryRegion
from .regions.code import CodeRegion
from .regions.vision import VisionRegion
from .regions.motor import MotorRegion
from .regions.emotional import EmotionalRegion


class PRDLLMBrain(nn.Module):
    """Complete Human-Like Brain Architecture"""
    
    def __init__(self, config: PRDLLMConfig):
        super().__init__()
        self.config = config
        
        # Create regions
        self.regions = nn.ModuleList([
            ReasoningRegion(0, config.d_model, config.d_ff),
            LanguageRegion(1, config.d_model, config.d_ff),
            MathRegion(2, config.d_model, config.d_ff),
            MemoryRegion(3, config.d_model, config.d_ff),
            CodeRegion(4, config.d_model, config.d_ff),
            VisionRegion(5, config.d_model, config.d_ff),
            MotorRegion(6, config.d_model, config.d_ff),
            EmotionalRegion(7, config.d_model, config.d_ff),
        ])
        
        # Router
        self.router = GlobalBrainRouter(
            d_model=config.d_model,
            num_regions=8,
            top_k=config.top_k,
            temperature=config.router_temp
        )
        
        # Dispatcher
        self.dispatcher = SparseDispatcher(
            d_model=config.d_model,
            num_regions=8,
            top_k=config.top_k
        )
        
        # Critic
        self.critic = CriticLayer(
            d_model=config.d_model,
            threshold=config.critic_threshold,
            hidden_dim=config.critic_hidden_dim
        )
        
        # Transformer components
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_drop = nn.Dropout(config.dropout)
        
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers // 2)
        ])
        
        # Output layers
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.output_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        
        self._prev_output = None
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                update_plasticity: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        
        B, T = input_ids.shape
        hidden = self.embed_drop(self.embed(input_ids))
        
        for block in self.shared_blocks:
            hidden = block(hidden)
        
        weights, indices, scores = self.router(hidden)
        dispatched, dispatch_stats = self.dispatcher(hidden, weights, indices, self.regions, update_plasticity)
        
        projected = self.output_proj(dispatched)
        normalized = self.output_norm(projected)
        raw_logits = self.lm_head(normalized)
        
        corrected_hidden, confidence, critic_stats = self.critic(normalized, self._prev_output)
        corrected_logits = self.lm_head(corrected_hidden)
        self._prev_output = corrected_hidden.detach()
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                corrected_logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        stats = {
            'router': self.router.get_stats(),
            'dispatcher': dispatch_stats,
            'critic': critic_stats,
            'confidence': confidence.mean().item(),
            'active_regions': dispatch_stats['active_regions'],
            'num_active': dispatch_stats['num_active'],
            'active_percentage': dispatch_stats['active_percentage'],
        }
        
        return corrected_logits, loss, stats
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        
        self.eval()
        generated = input_ids.clone()
        self._prev_output = None
        
        for step in range(max_new_tokens):
            ctx = generated[:, -self.config.max_seq_len:]
            logits, _, stats = self(ctx, update_plasticity=False)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == 2: # Assuming 2 is EOS
                break
        
        return generated
