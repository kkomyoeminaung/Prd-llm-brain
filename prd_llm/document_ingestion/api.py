"""
FastAPI endpoints for document ingestion
"""

import os
import tempfile
import shutil
from fastapi import UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel

from .parsers import DocumentIngestor, URLParser
from .knowledge_base import KnowledgeBase


class DocumentIngestionAPI:
    """API endpoints for document ingestion"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    async def upload_file(self, file: UploadFile) -> Dict:
        """Upload and process a single file"""
        # Save temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Parse and ingest
            text = DocumentIngestor.parse_file(tmp_path)
            
            if not text or len(text) < 20:
                raise HTTPException(status_code=400, detail="File contains no readable text")
            
            chunks = self.kb.add_file(file.filename, text)
            
            return {
                'status': 'success',
                'filename': file.filename,
                'chunks_added': chunks,
                'text_length': len(text)
            }
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def upload_url(self, url: str) -> Dict:
        """Fetch and ingest content from a URL"""
        text = await URLParser.fetch(url)
        
        if not text:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: {url}")
        
        chunks = self.kb.add_document(text, f"url:{url}", {'type': 'url', 'url': url})
        
        return {
            'status': 'success',
            'url': url,
            'chunks_added': chunks,
            'text_length': len(text)
        }
    
    async def ingest_text(self, text: str, source: str = "direct_input") -> Dict:
        """Direct text ingestion"""
        if not text or len(text) < 20:
            raise HTTPException(status_code=400, detail="Text too short")
        
        chunks = self.kb.add_document(text, source)
        
        return {
            'status': 'success',
            'source': source,
            'chunks_added': chunks,
            'text_length': len(text)
        }
    
    async def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.kb.get_stats()
