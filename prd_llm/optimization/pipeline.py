"""
Complete Model Optimization Pipeline
"""

import torch
import torch.nn as nn
import time
from typing import Dict

from .pruning import ModelPruner
from .quantization import ModelQuantizer
from .kv_cache import KVCache


class OptimizationPipeline:
    """
    Complete optimization pipeline for PRD-LLM
    """
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.optimized_model = None
        self.stats = {}
        
    def run_full_pipeline(self, 
                          prune_ratio: float = 0.3,
                          quantize: bool = True) -> nn.Module:
        print("\n🚀 STARTING OPTIMIZATION PIPELINE")
        
        model = self.original_model
        
        # Pruning
        pruner = ModelPruner(model)
        prune_stats = pruner.prune_model(sparsity_ratio=prune_ratio)
        self.stats['pruning'] = prune_stats
        
        # Quantization
        if quantize:
            quantizer = ModelQuantizer(model)
            model = quantizer.quantize_dynamic()
            self.stats['quantization'] = {'reduction': quantizer.reduction}
            
        self.optimized_model = model
        return model

    def get_stats(self) -> Dict:
        return self.stats
