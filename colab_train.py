import os
import torch
from prd_llm.config import PRDLLMConfig
from prd_llm.brain_model import PRDLLMBrain
from prd_llm.trainer import PRDTrainer, PRDDataset

def main():
    # 1. Setup config
    config = PRDLLMConfig()
    config.d_model = 256
    config.n_layers = 6
    config.batch_size = 16
    config.max_steps = 5000
    config.num_epochs = 5
    
    # 2. Create model
    model = PRDLLMBrain(config)
    
    # 3. Prepare data (Synthetic for demo)
    # In Colab, you would load your real tokens here
    print("Generating synthetic data...")
    dummy_data = [torch.randint(0, config.vocab_size, (128,)).tolist() for _ in range(100)]
    dataset = PRDDataset(dummy_data, max_seq_len=64)
    
    # 4. Train
    trainer = PRDTrainer(config)
    trained_model = trainer.train(model, dataset)
    
    # 5. Save
    trainer.save_checkpoint(trained_model, config.checkpoint_dir)
    print(f"Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()
