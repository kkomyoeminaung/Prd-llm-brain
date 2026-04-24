"""
PRD-LLM Inference Engine - Fast generation with brain-like architecture
"""

import torch
from typing import Optional, List, Dict

from .brain_model import PRDLLMBrain
from .tokenizer import PRDTokenizer
from .config import PRDLLMConfig


class PRDInferenceEngine:
    """Production inference engine for PRD-LLM Brain"""
    
    def __init__(self):
        self.model: Optional[PRDLLMBrain] = None
        self.tokenizer: Optional[PRDTokenizer] = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_ready = False
    
    def load(self, checkpoint_path: str, tokenizer_path: str, config: Optional[PRDLLMConfig] = None):
        """Load model and tokenizer from checkpoint"""
        print(f"[Inference] Loading model from {checkpoint_path}...")
        
        self.tokenizer = PRDTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        if config is None:
            config = PRDLLMConfig()
            config.num_regions = 4 # Adjusted for demo
            
        self.model = PRDLLMBrain(config)
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        except:
            print("Warning: Could not load weights, using randomized initialization.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_ready = True
        print(f"[Inference] Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> str:
        """Generate text from prompt"""
        if not self.is_ready:
            return "[Model not loaded]"
        
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        input_ids = input_ids.to(self.device)
        
        output_ids = self.model.generate(input_ids, max_new_tokens, temperature, top_k)
        output_text = self.tokenizer.decode(output_ids[0].tolist())
        
        return output_text
    
    def chat(self, messages: List[dict], max_new_tokens: int = 256,
             temperature: float = 0.8) -> str:
        """Chat interface"""
        if not self.is_ready:
            return "[Model not loaded]"
        
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant: "
        
        return self.generate(prompt, max_new_tokens, temperature)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        if self.model:
            return self.model.get_system_stats()
        return {'ready': False}
