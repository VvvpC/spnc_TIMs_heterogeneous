# -*- coding: utf-8 -*-
"""
Example usage of the modular SPNC evaluation framework.

This file demonstrates how to use the newly modularized evaluation system
for different types of reservoir computing performance assessments.
"""

# First, set up the environment
from config import setup_environment
setup_environment()

# Import the main components
from core import ReservoirParams
from framework import run_evaluation
import numpy as np

def example_single_parameter_scan():
    """Example: Single parameter scanning for Memory Capacity."""
    print("=== Example 1: Single Parameter Scan (MC) ===")
    
    # Create base reservoir parameters
    reservoir_params = ReservoirParams(
        beta_prime=20,
        Nvirt=30,
        m0=0.008,
        params={'theta': 0.3, 'gamma': 0.1}
    )
    
    # Run MC evaluation with beta_prime scanning
    result = run_evaluation(
        task_type='MC',
        param_name='beta_prime',
        param_range=[10, 20, 30, 50],
        reservoir_params=reservoir_params,
        reservoir_tag='example_mc_scan'
    )
    
    print(f"MC results: {result['MC']}")
    print()

def example_multi_parameter_grid():
    """Example: Multi-parameter grid search for KR&GR."""
    print("=== Example 2: Multi-Parameter Grid Search (KR&GR) ===")
    
    # Create base reservoir parameters
    reservoir_params = ReservoirParams(
        beta_prime=30,
        Nvirt=25,
        m0=0.008
    )
    
    # Define parameter grid
    param_grid = {
        'm0': np.linspace(0.005, 0.012, 4),
        'gamma': np.linspace(0.08, 0.15, 3)
    }
    
    # Run KR&GR evaluation
    result = run_evaluation(
        task_type='KRANDGR',
        param_grid=param_grid,
        reservoir_params=reservoir_params,
        reservoir_tag='example_kr_gr_grid',
        extra_args={'threshold': 0.05}
    )
    
    print(f"Grid search completed. {len(result['KR'])} combinations evaluated.")
    print(f"Best CQ: {max(result['CQ'])}")
    print()

def example_all_tasks():
    """Example: Run all evaluation tasks with single parameter set."""
    print("=== Example 3: All Tasks Evaluation ===")
    
    # Create optimized reservoir parameters
    reservoir_params = ReservoirParams(
        beta_prime=40,
        Nvirt=35,
        m0=0.009,
        params={'theta': 0.25, 'gamma': 0.12}
    )
    
    task_types = ['MC', 'KRANDGR', 'NARMA10', 'TI46']
    
    for task in task_types:
        print(f"\nRunning {task} evaluation...")
        
        try:
            result = run_evaluation(
                task_type=task,
                param_name='beta_prime',  # Single value scan
                param_range=[40],
                reservoir_params=reservoir_params,
                reservoir_tag=f'example_all_tasks_{task.lower()}'
            )
            
            # Print key results
            if task == 'MC':
                print(f"Memory Capacity: {result['MC'][0]:.3f}")
            elif task == 'KRANDGR':
                print(f"KR: {result['KR'][0]}, GR: {result['GR'][0]}, CQ: {result['CQ'][0]}")
            elif task == 'NARMA10':
                print(f"NARMA10 NRMSE: {result['NRMSE'][0]:.4f}")
            elif task == 'TI46':
                print(f"TI46 Accuracy: {result['acc'][0]:.3f}")
                
        except Exception as e:
            print(f"Error in {task}: {e}")

def example_custom_configuration():
    """Example: Custom configuration and validation."""
    print("=== Example 4: Custom Configuration ===")
    
    from framework.runner import get_available_tasks, validate_task_config
    
    # Show available tasks
    tasks = get_available_tasks()
    print("Available tasks:")
    for task_name, info in tasks.items():
        print(f"  {task_name}: {info['description']}")
    
    # Validate configuration
    validation = validate_task_config('MC', signal_len=300)
    print(f"\nValidation result: {validation}")
    
    # Create custom reservoir configuration
    custom_params = ReservoirParams()
    custom_params.update_params(
        beta_prime=25,
        h=0.5,
        Nvirt=40,
        theta=0.35,
        gamma=0.13
    )
    
    print("\nCustom parameters:")
    custom_params.print_params(verbose=True)

if __name__ == "__main__":
    print("SPNC Modular Evaluation Framework - Examples\n")
    
    try:
        example_single_parameter_scan()
        example_multi_parameter_grid()
        example_all_tasks()
        example_custom_configuration()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("Please ensure all dependencies are properly installed.")