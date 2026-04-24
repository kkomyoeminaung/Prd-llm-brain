"""
PRD-LLM: Complete Brain Architecture
"""

from .config import PRDLLMConfig, config
from .brain_model import PRDLLMBrain
from .tokenizer import PRDTokenizer
from .trainer import PRDTrainer
from .inference import PRDInferenceEngine

__version__ = "2.0.0"
__all__ = [
    'PRDLLMConfig',
    'config',
    'PRDLLMBrain',
    'PRDTokenizer',
    'PRDTrainer',
    'PRDInferenceEngine',
]
