# -*- coding: utf-8 -*-
"""
Test script for the modular SPNC evaluation framework.

This script performs basic functionality tests to ensure the modular
structure works correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing module imports...")
    
    try:
        # Test core imports
        from core import ReservoirParams, RunSpnc, MSE, NRMSE
        print("✓ Core modules imported successfully")
        
        # Test task imports  
        from tasks import evaluate_MC, evaluate_KRandGR, evaluate_NARMA10, evaluate_Ti46
        print("✓ Task modules imported successfully")
        
        # Test framework imports
        from framework import ReservoirPerformanceEvaluator, run_evaluation
        print("✓ Framework modules imported successfully")
        
        # Test config imports
        from config import setup_environment, get_config
        print("✓ Config modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_reservoir_params():
    """Test ReservoirParams functionality."""
    print("\nTesting ReservoirParams...")
    
    try:
        from core import ReservoirParams
        
        # Test basic creation
        params = ReservoirParams()
        print("✓ Default ReservoirParams created")
        
        # Test parameter update
        params.update_params(beta_prime=25, gamma=0.15)
        assert params.beta_prime == 25
        assert params.params['gamma'] == 0.15
        print("✓ Parameter updates work correctly")
        
        # Test custom initialization
        custom_params = ReservoirParams(
            Nvirt=40,
            m0=0.01,
            params={'theta': 0.3}
        )
        assert custom_params.Nvirt == 40
        assert custom_params.params['theta'] == 0.3
        print("✓ Custom initialization works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ ReservoirParams test failed: {e}")
        return False

def test_task_modules():
    """Test individual task module imports and basic structure."""
    print("\nTesting task modules...")
    
    try:
        # Test MC module
        from tasks.memory_capacity import generate_mc_signal, linear_memory_capacity
        from tasks.memory_capacity import evaluate_MC
        print("✓ Memory Capacity module structure OK")
        
        # Test KR&GR module
        from tasks.kr_gr import generate_kr_gr_input, evaluate_kr_gr_ranks
        from tasks.kr_gr import evaluate_KRandGR
        print("✓ KR&GR module structure OK")
        
        # Test signal generation
        import numpy as np
        signal = generate_mc_signal(100, washout=10, seed=1234)
        assert signal.shape == (90, 1)  # 100-10 washout samples
        print("✓ Signal generation works correctly")
        
        # Test KR&GR input generation  
        inputs = generate_kr_gr_input(5, 10, seed=1234)
        assert inputs.shape == (5, 20)  # 5 readouts, 10+10 columns
        print("✓ KR&GR input generation works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Task modules test failed: {e}")
        traceback.print_exc()
        return False

def test_framework_basic():
    """Test basic framework functionality without full evaluation."""
    print("\nTesting framework basics...")
    
    try:
        from core import ReservoirParams
        from framework import ReservoirPerformanceEvaluator
        from framework.runner import get_available_tasks, validate_task_config
        
        # Test task registry
        tasks = get_available_tasks()
        assert 'MC' in tasks
        assert 'KRANDGR' in tasks
        print("✓ Task registry accessible")
        
        # Test task validation
        validation = validate_task_config('MC')
        assert validation['valid'] == True
        print("✓ Task validation works")
        
        # Test evaluator creation (without running)
        params = ReservoirParams()
        
        # Mock task function for testing
        def mock_task(reservoir_params, **kwargs):
            return {'test_result': 1.0}
        
        evaluator = ReservoirPerformanceEvaluator(
            task=mock_task,
            param_name='beta_prime',
            param_range=[20, 30],
            result_keys=['test_result'],
            result_labels=['Test Result'],
            reservoir_params=params,
            reservoir_tag='test'
        )
        
        summary = evaluator.get_summary()
        assert summary['evaluation_mode'] == 'single-parameter'
        print("✓ Evaluator creation and configuration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Framework test failed: {e}")
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test environment configuration."""
    print("\nTesting environment setup...")
    
    try:
        from config import setup_environment, get_config
        
        # Test config creation
        config = get_config()
        assert config is not None
        print("✓ Config object accessible")
        
        # Test environment setup (may have warnings, but shouldn't crash)
        setup_environment()
        print("✓ Environment setup completed (check warnings above)")
        
        # Test validation
        validation = config.validate_environment()
        print(f"✓ Environment validation: {validation['status']}")
        print(f"  Available modules: {len(validation['available'])}")
        if validation['missing']:
            print(f"  Missing modules: {validation['missing']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("SPNC Modular Framework Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("ReservoirParams Tests", test_reservoir_params), 
        ("Task Module Tests", test_task_modules),
        ("Framework Basic Tests", test_framework_basic),
        ("Environment Setup Tests", test_environment_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular framework is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)