import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Ensure package is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prd_llm.inference import PRDInferenceEngine

app = FastAPI(title="PRD-LLM API")
engine = PRDInferenceEngine()

class Query(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

@app.on_event("startup")
async def startup_event():
    # Load defaults
    engine.load("./checkpoints/best_model.pt", "./tokenizer/tokenizer.json")

@app.post("/generate")
async def generate(query: Query):
    text = engine.generate(query.prompt, query.max_tokens, query.temperature)
    return {"text": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
