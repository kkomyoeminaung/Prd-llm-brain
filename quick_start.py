#!/usr/bin/env python3
"""
Quick Start Script - One-command setup and test
"""

import os
import sys
import torch
import argparse


def setup_environment():
    print("=" * 60)
    print("PRD-LLM Brain - Quick Start")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    for d in ['checkpoints', 'data', 'tokenizer', 'logs']:
        os.makedirs(d, exist_ok=True)
    
    print("Directories created")
    return True


def test_imports():
    print("\n[1/4] Testing imports...")
    try:
        from prd_llm.config import PRDLLMConfig
        from prd_llm.brain_model import PRDLLMBrain
        from prd_llm.tokenizer import PRDTokenizer
        from prd_llm.core.router import GlobalBrainRouter
        from prd_llm.core.plasticity import PlasticityAdapter
        from prd_llm.core.critic import CriticLayer
        print("  All imports successful")
        return True
    except Exception as e:
        print(f"  Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    print("\n[2/4] Testing model creation...")
    try:
        from prd_llm.config import PRDLLMConfig
        from prd_llm.brain_model import PRDLLMBrain
        
        config = PRDLLMConfig()
        config.d_model = 128
        config.n_layers = 4
        config.max_seq_len = 64
        
        model = PRDLLMBrain(config)
        print(f"  Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    print("\n[3/4] Testing forward pass...")
    try:
        from prd_llm.config import PRDLLMConfig
        from prd_llm.brain_model import PRDLLMBrain
        
        config = PRDLLMConfig()
        config.d_model = 128
        config.n_layers = 4
        config.max_seq_len = 64
        config.vocab_size = 1000
        
        model = PRDLLMBrain(config)
        model.eval()
        
        test_input = torch.randint(0, 1000, (1, 32))
        logits, loss, stats = model(test_input)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Active regions: {stats['active_regions']}")
        print(f"  Active percentage: {stats['active_percentage']:.1f}%")
        return True
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    print("\n[4/4] Testing generation...")
    try:
        from prd_llm.config import PRDLLMConfig
        from prd_llm.brain_model import PRDLLMBrain
        
        config = PRDLLMConfig()
        config.d_model = 128
        config.n_layers = 4
        config.max_seq_len = 64
        config.vocab_size = 1000
        
        model = PRDLLMBrain(config)
        model.eval()
        
        prompt = torch.randint(0, 1000, (1, 5))
        generated = model.generate(prompt, max_new_tokens=20)
        
        print(f"  Prompt shape: {prompt.shape}")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Generation successful")
        return True
    except Exception as e:
        print(f"  Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    setup_environment()
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_generation,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
            break
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" PRD-LLM Brain is ready for production!")
        print("=" * 60)
        print("\nTo start training:")
        print("  Edit colab_train.py with your data path and run it.")
    else:
        print(" Some tests failed. Please check dependencies.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
