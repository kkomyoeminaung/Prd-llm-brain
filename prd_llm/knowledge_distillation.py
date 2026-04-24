"""
Knowledge Distillation from Teacher AI (Gemini/Groq) to PRD-LLM
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DistilledSample:
    """Single distilled knowledge sample"""
    prompt: str
    response: str
    domain: str
    confidence: float
    source: str  # gemini, groq, wikipedia


class TeacherAI:
    """Base teacher AI interface"""
    
    async def generate(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        raise NotImplementedError


class GeminiTeacher(TeacherAI):
    """Google Gemini as teacher"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    async def generate(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System: {system_prompt}\n\nUser: {prompt}"}]})
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        body = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}?key={self.api_key}",
                    json=body, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[Gemini] Error: {e}")
        return None


class GroqTeacher(TeacherAI):
    """Groq (Llama) as teacher - fast and free"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def generate(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url, headers=headers, json=body,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Groq] Error: {e}")
        return None


class KnowledgeDistiller:
    """
    Distills knowledge from teacher AIs into training data for PRD-LLM
    """
    
    SYSTEM_PROMPT = """You are an expert teacher. Give clear, accurate, educational responses.
Be concise but thorough. Focus on facts and reasoning."""
    
    TOPICS = [
        "artificial intelligence", "machine learning", "neural networks",
        "physics", "quantum mechanics", "relativity", "thermodynamics",
        "biology", "genetics", "evolution", "neuroscience",
        "chemistry", "organic chemistry", "biochemistry",
        "mathematics", "calculus", "linear algebra", "statistics",
        "computer science", "programming", "algorithms", "data structures",
        "history", "philosophy", "psychology", "economics",
        "myanmar culture", "buddhism", "southeast asia history",
    ]
    
    QUESTION_TEMPLATES = [
        "Explain {topic} in simple terms.",
        "What are the key principles of {topic}?",
        "How does {topic} work?",
        "What is the history of {topic}?",
        "Why is {topic} important?",
        "What are common misconceptions about {topic}?",
        "Compare {topic} with related concepts.",
        "Give me a practical example of {topic}.",
    ]
    
    def __init__(self, output_path: str, teachers: List[TeacherAI]):
        self.output_path = output_path
        self.teachers = teachers
        self.total_distilled = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def _generate_prompts(self, n: int = 100) -> List[str]:
        """Generate diverse prompts for distillation"""
        import random
        prompts = []
        
        for _ in range(n):
            topic = random.choice(self.TOPICS)
            template = random.choice(self.QUESTION_TEMPLATES)
            prompt = template.format(topic=topic)
            prompts.append(prompt)
        
        return prompts
    
    async def run_distillation_cycle(self, n_prompts: int = 50):
        """Main distillation loop"""
        print(f"[Distiller] Starting distillation cycle: {n_prompts} prompts...")
        
        prompts = self._generate_prompts(n_prompts)
        samples = []
        
        for i, prompt in enumerate(prompts):
            for teacher in self.teachers:
                response = await teacher.generate(prompt, self.SYSTEM_PROMPT)
                if response and len(response) > 20:
                    samples.append(DistilledSample(
                        prompt=prompt,
                        response=response,
                        domain=self._detect_domain(prompt),
                        confidence=0.85,
                        source=teacher.__class__.__name__
                    ))
                    break
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_prompts} ({len(samples)} successful)")
            
            await asyncio.sleep(0.3)  # Rate limiting
        
        # Save to JSONL
        with open(self.output_path, 'a', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps({
                    'prompt': sample.prompt,
                    'response': sample.response,
                    'domain': sample.domain,
                    'confidence': sample.confidence,
                    'source': sample.source
                }, ensure_ascii=False) + '\n')
        
        self.total_distilled += len(samples)
        print(f"[Distiller] Saved {len(samples)} samples. Total: {self.total_distilled}")
        return samples
    
    def _detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ['physics', 'quantum', 'relativity']):
            return 'physics'
        if any(w in text_lower for w in ['biology', 'genetics', 'evolution']):
            return 'biology'
        if any(w in text_lower for w in ['math', 'calculus', 'algebra']):
            return 'mathematics'
        if any(w in text_lower for w in ['programming', 'algorithm', 'code']):
            return 'computer_science'
        return 'general'


def setup_distillation(api_keys: dict):
    """Setup distillation with available teachers"""
    teachers = []
    
    if api_keys.get('GEMINI_API_KEY'):
        teachers.append(GeminiTeacher(api_keys['GEMINI_API_KEY']))
        print("[Distiller] Gemini teacher loaded")
    
    if api_keys.get('GROQ_API_KEY'):
        teachers.append(GroqTeacher(api_keys['GROQ_API_KEY']))
        print("[Distiller] Groq teacher loaded")
    
    return teachers
