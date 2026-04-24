"""
Integration Tests for PRD-LLM
"""
import torch
import unittest
from prd_llm.brain_model import PRDLLMBrain
from prd_llm.config import PRDLLMConfig

class TestModelIntegration(unittest.TestCase):
    def setUp(self):
        config = PRDLLMConfig()
        config.d_model = 128
        config.n_layers = 2
        self.model = PRDLLMBrain(config)
        
    def test_full_forward(self):
        input_ids = torch.randint(0, 1000, (1, 16))
        logits, _, stats = self.model(input_ids)
        self.assertIn('active_regions', stats)
        self.assertEqual(logits.shape[0], 1)

def run_integration_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestModelIntegration)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result.wasSuccessful()
