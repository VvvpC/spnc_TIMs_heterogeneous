# -*- coding: utf-8 -*-
"""
Basic test script for the modular SPNC evaluation framework.

This script tests the modular structure without requiring all dependencies.
"""

import sys
import os

def test_directory_structure():
    """Test that the directory structure was created correctly."""
    print("Testing directory structure...")
    
    required_dirs = [
        'core',
        'tasks',
        'tasks/memory_capacity', 
        'tasks/kr_gr',
        'tasks/narma10',
        'tasks/ti46',
        'framework',
        'config'
    ]
    
    required_files = [
        'core/__init__.py',
        'core/reservoir.py',
        'core/base_utils.py',
        'tasks/__init__.py',
        'tasks/memory_capacity/__init__.py',
        'tasks/memory_capacity/signals.py',
        'tasks/memory_capacity/processing.py',
        'tasks/memory_capacity/evaluator.py',
        'framework/__init__.py',
        'framework/evaluator.py',
        'framework/runner.py',
        'config/__init__.py',
        'config/paths.py'
    ]
    
    missing = []
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(f"Directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(f"File: {file_path}")
    
    if missing:
        print("FAIL - Missing components:")
        for item in missing:
            print(f"  - {item}")
        return False
    else:
        print("PASS - All required components exist")
        return True

def test_basic_imports():
    """Test basic Python imports without external dependencies."""
    print("\nTesting basic imports...")
    
    try:
        # Test basic utility functions
        import sys
        sys.path.insert(0, '.')
        
        # Test core utilities (should work without external deps)
        from core.base_utils import MSE, NRMSE, safe_numpy_convert
        print("PASS - Core utilities imported")
        
        # Test that files contain expected functions
        import inspect
        from tasks.memory_capacity.signals import generate_mc_signal
        assert callable(generate_mc_signal)
        print("PASS - Task functions are callable")
        
        # Test framework structure
        from framework.runner import TASK_REGISTRY, get_available_tasks
        tasks = get_available_tasks()
        assert 'MC' in tasks
        assert 'KRANDGR' in tasks
        print("PASS - Framework structure valid")
        
        return True
        
    except Exception as e:
        print(f"FAIL - Import error: {e}")
        return False

def test_parameter_class():
    """Test ReservoirParams class without external dependencies."""
    print("\nTesting ReservoirParams class...")
    
    try:
        # Mock the external dependencies
        import sys
        import types
        
        # Create mock modules
        mock_spnc = types.ModuleType('spnc')
        mock_spnc.spnc_anisotropy = lambda *args, **kwargs: None
        sys.modules['spnc'] = mock_spnc
        
        mock_single_node_res = types.ModuleType('single_node_res')
        mock_single_node_res.single_node_reservoir = lambda *args, **kwargs: None
        sys.modules['single_node_res'] = mock_single_node_res
        
        mock_deterministic_mask = types.ModuleType('deterministic_mask')
        mock_deterministic_mask.fixed_seed_mask = lambda *args, **kwargs: None
        mock_deterministic_mask.max_sequences_mask = lambda *args, **kwargs: None
        sys.modules['deterministic_mask'] = mock_deterministic_mask
        
        # Now test ReservoirParams
        from core.reservoir import ReservoirParams
        
        # Test default creation
        params = ReservoirParams()
        assert hasattr(params, 'h')
        assert hasattr(params, 'beta_prime')
        assert hasattr(params, 'Nvirt')
        assert isinstance(params.params, dict)
        print("PASS - ReservoirParams default creation")
        
        # Test parameter updates
        params.update_params(beta_prime=50, gamma=0.2)
        assert params.beta_prime == 50
        assert params.params['gamma'] == 0.2
        print("PASS - Parameter updates work")
        
        # Test copy
        params_copy = params.copy()
        assert params_copy.beta_prime == 50
        assert params_copy is not params
        print("PASS - Parameter copy works")
        
        return True
        
    except Exception as e:
        print(f"FAIL - ReservoirParams test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_structure():
    """Test task module structure and organization."""
    print("\nTesting task module structure...")
    
    try:
        # Test memory capacity structure
        from tasks.memory_capacity.signals import generate_mc_signal, linear_memory_capacity
        from tasks.memory_capacity.processing import ridge_regression_for_mc
        
        # Test basic functionality
        import numpy as np
        signal = generate_mc_signal(100, washout=20, seed=1234)
        assert signal.shape == (80, 1)
        print("PASS - Memory capacity signal generation")
        
        # Test KR&GR structure
        from tasks.kr_gr.signals import generate_kr_gr_input
        from tasks.kr_gr.processing import evaluate_kr_gr_ranks
        
        inputs = generate_kr_gr_input(5, 8, seed=1234) 
        assert inputs.shape == (5, 18)  # 5 readouts, 8+10 columns
        print("PASS - KR&GR input generation")
        
        # Test that evaluators exist
        from tasks.memory_capacity.evaluator import evaluate_MC
        from tasks.kr_gr.evaluator import evaluate_KRandGR
        from tasks.narma10.evaluator import evaluate_NARMA10
        from tasks.ti46.evaluator import evaluate_Ti46
        
        print("PASS - All evaluator functions exist")
        
        return True
        
    except Exception as e:
        print(f"FAIL - Task structure error: {e}")
        return False

def test_framework_logic():
    """Test framework logic without running full evaluations."""
    print("\nTesting framework logic...")
    
    try:
        from framework.evaluator import ReservoirPerformanceEvaluator
        from core.reservoir import ReservoirParams
        
        # Mock reservoir params
        params = ReservoirParams()
        
        # Mock task function
        def mock_task(reservoir_params, **kwargs):
            return {'result': reservoir_params.beta_prime * 0.1}
        
        # Test single parameter evaluator
        evaluator = ReservoirPerformanceEvaluator(
            task=mock_task,
            param_name='beta_prime',
            param_range=[10, 20, 30],
            result_keys=['result'],
            result_labels=['Mock Result'],
            reservoir_params=params,
            reservoir_tag='test'
        )
        
        assert not evaluator.is_multi_param
        assert evaluator.param_name == 'beta_prime'
        print("PASS - Single parameter evaluator setup")
        
        # Test multi-parameter evaluator
        multi_evaluator = ReservoirPerformanceEvaluator(
            task=mock_task,
            param_grid={'beta_prime': [20, 30], 'h': [0.3, 0.4]},
            result_keys=['result'],
            result_labels=['Mock Result'],
            reservoir_params=params,
            reservoir_tag='test_multi'
        )
        
        assert multi_evaluator.is_multi_param
        assert len(multi_evaluator.param_combinations) == 4  # 2x2 grid
        print("PASS - Multi-parameter evaluator setup")
        
        return True
        
    except Exception as e:
        print(f"FAIL - Framework logic error: {e}")
        return False

def run_basic_tests():
    """Run all basic tests."""
    print("=" * 50)
    print("SPNC Modular Framework - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Basic Imports", test_basic_imports),
        ("Parameter Class", test_parameter_class), 
        ("Task Structure", test_task_structure),
        ("Framework Logic", test_framework_logic)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"CRASH - {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSuccess! The modular framework structure is correct.")
        print("You can now use the framework by importing:")
        print("  from core import ReservoirParams")
        print("  from framework import run_evaluation")
        return True
    else:
        print("\nSome tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)