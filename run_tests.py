"""
Complete Test Runner for PRD-LLM
"""
from tests.test_unit import run_unit_tests
from tests.test_integration import run_integration_tests
from tests.test_stress import run_stress_tests
from tests.test_benchmark import run_benchmarks

def run_all_tests():
    print("--- Starting Test Suite ---")
    unit = run_unit_tests()
    inter = run_integration_tests()
    stress = run_stress_tests()
    bench = run_benchmarks()
    
    return {
        "unit": {"status": "passed" if unit else "failed"},
        "integration": {"status": "passed" if inter else "failed"},
        "stress": stress,
        "benchmark": bench,
        "all_passed": unit and inter
    }
