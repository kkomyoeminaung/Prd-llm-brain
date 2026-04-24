"""
PRD-LLM Trainer - Training loop with plasticity and self-correction
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm

from .brain_model import PRDLLMBrain
from .config import PRDLLMConfig


class PRDDataset(Dataset):
    """Simple dataset for training"""
    def __init__(self, data_samples: List[List[int]], max_seq_len: int = 128):
        self.samples = data_samples
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        max_len = self.max_seq_len
        
        if len(tokens) <= max_len:
            x = tokens[:-1]
            y = tokens[1:]
            pad_len = max_len - len(x)
            x = x + [0] * pad_len
            y = y + [-1] * pad_len
        else:
            x = tokens[:max_len]
            y = tokens[1:max_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PRDTrainer:
    """Trainer for PRDLLMBrain"""
    
    def __init__(self, config: PRDLLMConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.step = 0
        self.best_loss = float('inf')
    
    def train(self, model: PRDLLMBrain, train_dataset: PRDDataset,
              val_dataset: Optional[PRDDataset] = None) -> PRDLLMBrain:
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        model.train()
        model.to(self.device)
        
        print(f"[Trainer] Starting training on {self.device}")
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                logits, loss, stats = model(x, y, update_plasticity=True)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                self.step += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if self.step >= self.config.max_steps:
                    break
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            if val_dataset:
                val_loss = self.evaluate(model, val_dataset)
                print(f"Validation Loss: {val_loss:.4f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(model, self.config.checkpoint_dir)
            
            if self.step >= self.config.max_steps:
                break
        
        return model
    
    @torch.no_grad()
    def evaluate(self, model: PRDLLMBrain, val_dataset: PRDDataset) -> float:
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        losses = []
        
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, loss, _ = model(x, y, update_plasticity=False)
            losses.append(loss.item())
        
        model.train()
        return np.mean(losses)
    
    def save_checkpoint(self, model: PRDLLMBrain, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "best_model.pt"))
