"""
PRD-LLM with RAG (Retrieval-Augmented Generation)
"""

from typing import List, Dict
import torch
from .document_ingestion.knowledge_base import KnowledgeBase

class RAG_PRDLLM:
    """
    PRD-LLM with document knowledge base
    """
    
    def __init__(self, model, tokenizer, knowledge_base_path: str = "./knowledge_base"):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
    
    def generate_with_context(self, query: str, max_new_tokens: int = 300, 
                              temperature: float = 0.8, use_rag: bool = True) -> Dict:
        """
        Generate response with relevant context from knowledge base
        """
        # Search for relevant knowledge
        context = ""
        if use_rag:
            context = self.knowledge_base.get_context_for_prompt(query)
        
        # Build prompt with context
        if context:
            prompt = f"{context}\n\nBased on the above information, please answer the following question:\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        # In simulation mode, model might be None
        if self.model is None:
            return {
                "text": f"[RAG Simulation] Response to: {query}",
                "context_used": context != ""
            }
        
        # Real generation logic
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        # Move to model device
        if hasattr(self.model, 'parameters'):
            input_ids = input_ids.to(next(self.model.parameters()).device)
        
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        response = self.tokenizer.decode(output_ids[0].tolist())
        
        # Extract answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return {
            "text": response,
            "context_used": context != ""
        }
