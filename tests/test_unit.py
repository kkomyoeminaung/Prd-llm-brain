"""
Unit Tests for PRD-LLM Components
"""
import torch
import unittest
from prd_llm.core.router import GlobalBrainRouter
from prd_llm.core.plasticity import PlasticityAdapter
from prd_llm.core.critic import CriticLayer

class TestRouter(unittest.TestCase):
    def setUp(self):
        self.router = GlobalBrainRouter(d_model=128, num_regions=8, top_k=2)
    
    def test_routing_shape(self):
        x = torch.randn(2, 16, 128)
        weights, indices, _ = self.router(x)
        self.assertEqual(indices.shape[-1], 2)
        self.assertEqual(weights.shape[-1], 8)

class TestPlasticity(unittest.TestCase):
    def setUp(self):
        self.plasticity = PlasticityAdapter(d_model=64)
        
    def test_forward_output(self):
        x = torch.randn(2, 8, 64)
        out = self.plasticity(x)
        self.assertEqual(out.shape, x.shape)

class TestCritic(unittest.TestCase):
    def setUp(self):
        self.critic = CriticLayer(d_model=64)
        
    def test_confidence_range(self):
        x = torch.randn(2, 16, 64)
        _, conf, _ = self.critic(x)
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())

def run_unit_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestPlasticity))
    suite.addTests(loader.loadTestsFromTestCase(TestCritic))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result.wasSuccessful()
