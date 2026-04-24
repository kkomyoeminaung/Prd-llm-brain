"""
Vector database for document storage and retrieval
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class KnowledgeEntry:
    """Single knowledge entry"""
    id: str
    content: str
    source: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SimpleVectorStore:
    """
    Simple vector store for document embeddings
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.entries: List[KnowledgeEntry] = []
    
    def add(self, content: str, source: str, embedding: List[float], metadata: Dict = None):
        """Add a knowledge entry"""
        entry_id = f"{source}_{len(self.entries)}"
        self.entries.append(KnowledgeEntry(
            id=entry_id,
            content=content,
            source=source,
            embedding=embedding,
            metadata=metadata or {}
        ))
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """Search for similar entries using cosine similarity"""
        if not self.entries:
            return []
        
        similarities = []
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        for entry in self.entries:
            if not entry.embedding:
                continue
            entry_vec = np.array(entry.embedding)
            entry_norm = np.linalg.norm(entry_vec)
            if entry_norm == 0 or query_norm == 0:
                continue
            similarity = np.dot(query_vec, entry_vec) / (query_norm * entry_norm)
            similarities.append((entry, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, path: str):
        """Save to file"""
        data = []
        for entry in self.entries:
            data.append({
                'id': entry.id,
                'content': entry.content,
                'source': entry.source,
                'embedding': entry.embedding,
                'metadata': entry.metadata
            })
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load from file"""
        if not os.path.exists(path): return
        with open(path, 'r') as f:
            data = json.load(f)
        self.entries = []
        for item in data:
            self.entries.append(KnowledgeEntry(
                id=item['id'],
                content=item['content'],
                source=item['source'],
                embedding=item['embedding'],
                metadata=item.get('metadata', {})
            ))


class SimpleEmbedder:
    """Simple embedder for text using hashing (simulated)"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed(self, text: str) -> List[float]:
        """Convert text to embedding vector (Simulated)"""
        import hashlib
        
        words = text.lower().split()
        embedding = [0.0] * self.dimension
        
        for word in words[:200]:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dimension
            embedding[idx] += 1.0
        
        # Normalize
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class KnowledgeBase:
    """
    Complete knowledge base for document storage and retrieval
    """
    
    def __init__(self, storage_path: str = "./knowledge_base"):
        self.storage_path = storage_path
        self.vector_store = SimpleVectorStore()
        self.embedder = SimpleEmbedder()
        self.document_sources = set()
        
        os.makedirs(storage_path, exist_ok=True)
        self._load()
    
    def add_document(self, content: str, source: str, metadata: Dict = None):
        """Add a document to knowledge base"""
        from .parsers import DocumentIngestor
        chunks = DocumentIngestor.chunk_text(content, chunk_size=500)
        embeddings = self.embedder.embed_batch(chunks)
        
        for chunk, embedding in zip(chunks, embeddings):
            self.vector_store.add(chunk, source, embedding, metadata)
        
        self.document_sources.add(source)
        self._save()
        
        return len(chunks)
    
    def add_file(self, filename: str, content: str):
        """Add file content to knowledge base"""
        return self.add_document(content, f"file:{filename}", {'type': 'file', 'filename': filename})
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant knowledge"""
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return [{
            'content': entry.content,
            'source': entry.source,
            'similarity': float(sim),
            'metadata': entry.metadata
        } for entry, sim in results]
    
    def get_context_for_prompt(self, query: str, max_chunks: int = 3) -> str:
        """Get relevant context to inject into prompt"""
        results = self.search(query, top_k=max_chunks)
        
        if not results:
            return ""
        
        context = "Relevant information from knowledge base:\n\n"
        for i, r in enumerate(results):
            context += f"[{i+1}] From {r['source']}:\n{r['content']}\n\n"
        
        return context
    
    def get_stats(self) -> Dict:
        return {
            'total_entries': len(self.vector_store.entries),
            'document_sources': list(self.document_sources),
        }
    
    def _save(self):
        self.vector_store.save(f"{self.storage_path}/vectors.json")
        with open(f"{self.storage_path}/sources.json", 'w') as f:
            json.dump(list(self.document_sources), f)
    
    def _load(self):
        self.vector_store.load(f"{self.storage_path}/vectors.json")
        sources_path = f"{self.storage_path}/sources.json"
        if os.path.exists(sources_path):
            with open(sources_path, 'r') as f:
                self.document_sources = set(json.load(f))
