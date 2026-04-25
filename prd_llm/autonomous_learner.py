"""
Complete Autonomous Learning System
Integrates: Knowledge Distillation + Internet Self-Learning + Dream Mode
"""

import os
import json
import asyncio
import threading
import time
from typing import Dict, Optional
from datetime import datetime

from .knowledge_distillation import KnowledgeDistiller, setup_distillation
from .self_learner import InternetSelfLearner
from .dream_mode import DreamMode, DreamAwareAPI


class AutonomousLearner:
    """
    Complete autonomous learning system for PRD-LLM
    - Learns from teacher AI (distillation)
    - Learns from internet (self-learning)
    - Consolidates during idle (dream mode)
    """
    
    def __init__(self, model, tokenizer, device, data_dir: str = "./data"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.distillation_path = f"{data_dir}/distilled.jsonl"
        self.selflearn_path = f"{data_dir}/selflearned.jsonl"
        self.db_path = f"{data_dir}/learner.db"
        
        # Teacher setup (will be initialized with API keys)
        self.distiller = None
        self.self_learner = None
        self.dream_api = None
        
        # Learning schedule
        self.last_distillation = None
        self.last_self_learn = None
        
    def setup_teachers(self, api_keys: dict):
        """Setup teacher AIs with API keys"""
        teachers = []
        
        if api_keys.get('GEMINI_API_KEY'):
            from .knowledge_distillation import GeminiTeacher
            teachers.append(GeminiTeacher(api_keys['GEMINI_API_KEY']))
            print("[AutoLearner] Gemini teacher ready")
        
        if api_keys.get('GROQ_API_KEY'):
            from .knowledge_distillation import GroqTeacher
            teachers.append(GroqTeacher(api_keys['GROQ_API_KEY']))
            print("[AutoLearner] Groq teacher ready")
        
        if teachers:
            self.distiller = KnowledgeDistiller(self.distillation_path, teachers)
        
        # Self-learner needs a teacher for QA extraction
        teacher = teachers[0] if teachers else None
        self.self_learner = InternetSelfLearner(self.selflearn_path, self.db_path, teacher)
        
    def setup_dream_mode(self):
        """Setup dream mode for idle learning"""
        self.dream_api = DreamAwareAPI(self.model, self.tokenizer, self.device, enable_dream_mode=True)
        print("[AutoLearner] Dream mode ready")
    
    async def distill_knowledge(self, n_prompts: int = 50):
        """Run knowledge distillation from teacher AI"""
        if not self.distiller:
            print("[AutoLearner] No teacher AI configured")
            return 0
        
        print(f"\n📖 [AutoLearner] Starting knowledge distillation...")
        samples = await self.distiller.run_distillation_cycle(n_prompts)
        self.last_distillation = datetime.now()
        
        # Auto-train on distilled data
        if samples:
            await self._train_on_new_data(self.distillation_path)
        
        return len(samples)
    
    async def learn_from_internet(self, max_topics: int = 3):
        """Run internet self-learning"""
        if not self.self_learner:
            print("[AutoLearner] Self-learner not configured")
            return 0
        
        print(f"\n🌐 [AutoLearner] Starting internet self-learning...")
        results = await self.self_learner.run_learning_cycle(max_topics)
        self.last_self_learn = datetime.now()
        
        # Auto-train on new data
        if results['total_qa'] > 0:
            await self._train_on_new_data(self.selflearn_path)
        
        return results['total_qa']
    
    async def _train_on_new_data(self, data_path: str):
        """Fine-tune model on new data (Real Weight Updates)"""
        print(f"[AutoLearner] Training on new data from {data_path}")
        
        if not os.path.exists(data_path):
            return
            
        # Load samples
        samples = []
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except: continue
        
        if not samples: return
        
        # Take latest 50 samples for quick fine-tuning
        recent_samples = samples[-50:]
        
        # Setup real optimizer
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        print(f"  🧠 Updating neural weights on {len(recent_samples)} samples...")
        
        for sample in recent_samples:
            # We assume sample has 'prompt' and 'response' or is structured for training
            text = f"{sample.get('prompt', '')} {sample.get('response', '')}"
            if not text.strip(): continue
            
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 5: continue
            
            input_ids = torch.tensor([tokens], device=self.device)
            targets = input_ids.clone()
            
            # Simple single-step SGD
            optimizer.zero_grad()
            logits, loss, stats = self.model(input_ids, targets=targets)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
        
        self.model.eval()
        print(f"  ✅ Neural Plasticity Active: Weights updated.")
    
    async def full_learning_cycle(self):
        """Complete autonomous learning cycle"""
        print("\n" + "="*60)
        print("🔄 AUTONOMOUS LEARNING CYCLE STARTED")
        print("="*60)
        
        # Step 1: Knowledge Distillation from Teachers
        distilled = await self.distill_knowledge(n_prompts=30)
        print(f"  📖 Distilled: {distilled} samples")
        
        # Step 2: Internet Self-Learning
        learned = await self.learn_from_internet(max_topics=3)
        print(f"  🌐 Self-learned: {learned} samples")
        
        # Step 3: Dream mode will consolidate during idle
        print(f"  💭 Dream mode will consolidate during idle")
        
        print("="*60)
        print("✅ LEARNING CYCLE COMPLETE")
        print("="*60)
        
        return {'distilled': distilled, 'self_learned': learned}
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        stats = {
            'dream_stats': self.dream_api.get_dream_stats() if self.dream_api else {},
            'self_learn_stats': self.self_learner.get_stats() if self.self_learner else {},
        }
        
        if self.last_distillation:
            stats['last_distillation'] = self.last_distillation.isoformat()
        if self.last_self_learn:
            stats['last_self_learn'] = self.last_self_learn.isoformat()
        
        return stats
    
    def start_background_learning(self, interval_hours: int = 6):
        """Start background learning thread"""
        def background_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            while True:
                time.sleep(interval_hours * 3600)
                loop.run_until_complete(self.full_learning_cycle())
        
        thread = threading.Thread(target=background_loop, daemon=True)
        thread.start()
        print(f"[AutoLearner] Background learning every {interval_hours} hours")
    
    def shutdown(self):
        """Shutdown all components"""
        if self.dream_api:
            self.dream_api.shutdown()
