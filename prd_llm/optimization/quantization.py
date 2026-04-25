"""
Model Quantization - Convert FP32 to INT8/INT4
Reduces memory by 75-87.5% while preserving accuracy
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ModelQuantizer:
    """
    Quantize model to lower precision
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._get_model_size_mb()
        self.reduction = 0
        
    def _get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def quantize_dynamic(self) -> nn.Module:
        """
        Dynamic quantization (INT8 for Linear layers)
        """
        print("\n🔢 Applying dynamic quantization (FP32 → INT8)...")
        
        try:
            # Check if running on CPU or if dynamic quantization is supported
            # In a real environment, you'd target specific platforms
            if hasattr(torch, "ao") and hasattr(torch.ao, "quantization"):
                quantized_model = torch.ao.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            else:
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            
            new_size = self._get_model_size_mb()
            self.reduction = (1 - new_size / self.original_size) * 100 if self.original_size > 0 else 0
            
            return quantized_model
        except Exception as e:
            print(f"   ⚠️ Dynamic quantization failed or unsupported in this env: {e}")
            return self.model
