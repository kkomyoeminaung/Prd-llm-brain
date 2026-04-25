import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

@dataclass
class Experience:
    """Single experience for dream learning"""
    input_text: str
    output_text: str
    confidence: float
    active_regions: List[int]
    timestamp: float

class ExperienceBuffer:
    """Buffer for storing experiences during inference"""
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.low_confidence_buffer = deque(maxlen=500)
        
    def add(self, input_text: str, output_text: str, confidence: float, active_regions: List[int]):
        exp = Experience(
            input_text=input_text,
            output_text=output_text,
            confidence=confidence,
            active_regions=active_regions,
            timestamp=time.time()
        )
        self.buffer.append(exp)
        if confidence < 0.5:
            self.low_confidence_buffer.append(exp)
    
    def get_low_confidence_experiences(self, count: int = 10) -> List[Experience]:
        return list(self.low_confidence_buffer)[-count:]
    
    def get_high_confidence_experiences(self, count: int = 10) -> List[Experience]:
        high_conf = [e for e in self.buffer if e.confidence > 0.7]
        return high_conf[-count:] if high_conf else []
    
    def clear_low_confidence(self):
        self.low_confidence_buffer.clear()
    
    def get_stats(self) -> Dict:
        return {
            'total_experiences': len(self.buffer),
            'low_confidence_count': len(self.low_confidence_buffer),
            'avg_confidence': sum(e.confidence for e in self.buffer) / len(self.buffer) if self.buffer else 0
        }

class DreamMode:
    """Consolidates learning from experiences during idle time."""
    def __init__(self, model, tokenizer, device, dream_interval_seconds: int = 60):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dream_interval = dream_interval_seconds
        self.experience_buffer = ExperienceBuffer()
        
        self.is_dreaming = False
        self.dream_count = 0
        self.total_corrections = 0
        self.running = False
        self.thread = None
        
    def collect_experience(self, input_text: str, output_text: str, confidence: float, active_regions: List[int]):
        self.experience_buffer.add(input_text, output_text, confidence, active_regions)

    def _dream_cycle(self):
        """Real weight consolidation during dream state"""
        print(f"[DreamMode] Cycle #{self.dream_count + 1} starting...")
        self.is_dreaming = True
        
        # Consolidate low confidence memories
        exps = self.experience_buffer.get_low_confidence_experiences(10)
        if exps:
            print(f"  💭 Dreaming: Consolidating {len(exps)} uncertain memories...")
            self.model.train()
            # Lower learning rate for consolidation
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)
            
            for exp in exps:
                text = f"{exp.input_text} {exp.output_text}"
                tokens = self.tokenizer.encode(text)
                if not tokens: continue
                
                input_ids = torch.tensor([tokens], device=self.device)
                optimizer.zero_grad()
                logits, loss, _ = self.model(input_ids, targets=input_ids)
                if loss is not None:
                    loss.backward()
                    optimizer.step()
            
            self.model.eval()
            self.total_corrections += len(exps)
            self.experience_buffer.clear_low_confidence()
            
        self.dream_count += 1
        self.is_dreaming = False
        print(f"[DreamMode] Cycle #{self.dream_count} completed.")

    def _loop(self):
        while self.running:
            time.sleep(self.dream_interval)
            if not self.is_dreaming:
                self._dream_cycle()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def get_stats(self):
        return {
            "is_dreaming": self.is_dreaming,
            "dream_count": self.dream_count,
            "total_corrections": self.total_corrections,
            "buffer_stats": self.experience_buffer.get_stats()
        }

class DreamAwareAPI:
    """Wrapper around DreamMode to provide easy access for APIs and AutonomousLearner"""
    def __init__(self, model, tokenizer, device, enable_dream_mode: bool = True):
        self.dream_mode = DreamMode(model, tokenizer, device)
        if enable_dream_mode:
            self.dream_mode.start()
            
    def collect_experience(self, input_text: str, output_text: str, confidence: float, active_regions: List[int]):
        self.dream_mode.collect_experience(input_text, output_text, confidence, active_regions)
        
    def get_dream_stats(self) -> Dict:
        return self.dream_mode.get_stats()
        
    def shutdown(self):
        self.dream_mode.stop()
