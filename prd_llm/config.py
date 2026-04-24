"""
Configuration for PRD-LLM Brain Architecture
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRDLLMConfig:
    """Complete configuration for brain-like model"""
    
    # Architecture
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Brain Router (Sparse Activation)
    num_regions: int = 8
    top_k: int = 2
    router_temp: float = 0.8
    
    # Plasticity (Hebbian Learning)
    plasticity_lr: float = 0.01
    plasticity_decay: float = 0.99
    plasticity_update_freq: int = 10
    
    # Critic (Self-Correction)
    critic_threshold: float = 0.5
    max_correction_loops: int = 3
    critic_hidden_dim: int = 256
    
    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    max_steps: int = 50000
    weight_decay: float = 0.1
    num_epochs: int = 10
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./data"
    tokenizer_path: str = "./tokenizer/tokenizer.json"
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.top_k <= self.num_regions


config = PRDLLMConfig()
