# -*- coding: utf-8 -*-
"""
Unified evaluation runner interface.

This module provides the run_evaluation function which serves as the
main entry point for executing reservoir performance evaluations.
"""

from core.reservoir import ReservoirParams
from tasks import evaluate_MC, evaluate_KRandGR, evaluate_NARMA10, evaluate_Ti46
from .evaluator import ReservoirPerformanceEvaluator

# Task registry mapping task types to evaluation functions and metadata
TASK_REGISTRY = {
    'MC': {
        'function': evaluate_MC,
        'result_keys': ['MC'],
        'result_labels': ['Memory Capacity'],
        'description': 'Linear memory capacity evaluation using delayed signal reconstruction'
    },
    'KRANDGR': {
        'function': evaluate_KRandGR,
        'result_keys': ['KR', 'GR', 'CQ'],
        'result_labels': ['Kernel Rank', 'Generalization Rank', 'Computational Quality'],
        'description': 'Computational quality evaluation through rank analysis'
    },
    'KR_GR': {  # Alternative name
        'function': evaluate_KRandGR,
        'result_keys': ['KR', 'GR', 'CQ'],
        'result_labels': ['Kernel Rank', 'Generalization Rank', 'Computational Quality'],
        'description': 'Computational quality evaluation through rank analysis'
    },
    'KR&GR': {  # Alternative name
        'function': evaluate_KRandGR,
        'result_keys': ['KR', 'GR', 'CQ'],
        'result_labels': ['Kernel Rank', 'Generalization Rank', 'Computational Quality'],
        'description': 'Computational quality evaluation through rank analysis'
    },
    'NARMA10': {
        'function': evaluate_NARMA10,
        'result_keys': ['NRMSE', 'y_test', 'pred'],
        'result_labels': ['NRMSE', 'Desired Output', 'Predicted Output'],
        'description': 'NARMA10 nonlinear system identification benchmark'
    },
    'TI46': {
        'function': evaluate_Ti46,
        'result_keys': ['acc'],
        'result_labels': ['Accuracy'],
        'description': 'TI46 digit recognition speech processing task'
    }
}

def run_evaluation(task_type, param_name=None, param_range=None, param_grid=None,
                  reservoir_params=None, result_dir="./Res_Tasks_Results", 
                  extra_args=None, reservoir_tag='default'):
    """
    Run performance evaluation for a specified task with parameter scanning.
    
    This is the main entry point for conducting reservoir performance evaluations.
    It supports both single-parameter scanning and multi-parameter grid search.
    
    Args:
        task_type (str): Type of evaluation task ('MC', 'KRANDGR', 'NARMA10', 'TI46')
        param_name (str): Name of parameter to scan (for single-parameter mode)
        param_range (list): Values to test for single parameter
        param_grid (dict): Multi-parameter grid {param_name: [values]}
        reservoir_params (ReservoirParams): Base reservoir configuration
        result_dir (str): Directory to save results
        extra_args (dict): Additional arguments passed to evaluation function
        reservoir_tag (str): Identifier for this evaluation run
    
    Returns:
        dict: Complete evaluation results
    
    Raises:
        ValueError: If task_type is not recognized
        
    Examples:
        # Single parameter scanning
        >>> result = run_evaluation(
        ...     task_type='MC',
        ...     param_name='beta_prime',
        ...     param_range=[10, 20, 30, 50],
        ...     reservoir_params=ReservoirParams(),
        ...     reservoir_tag='mc_beta_scan'
        ... )
        
        # Multi-parameter grid search
        >>> result = run_evaluation(
        ...     task_type='KR&GR',
        ...     param_grid={'m0': [0.005, 0.008, 0.010], 'gamma': [0.1, 0.15, 0.2]},
        ...     reservoir_params=ReservoirParams(beta_prime=30),
        ...     reservoir_tag='kr_gr_grid_search'
        ... )
    """
    # Validate and normalize task type
    task_key = task_type.upper()
    if task_key not in TASK_REGISTRY:
        available_tasks = list(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task_type: {task_type}. Available tasks: {available_tasks}")
    
    # Get task configuration
    task_config = TASK_REGISTRY[task_key]
    
    # Use default reservoir params if none provided
    if reservoir_params is None:
        reservoir_params = ReservoirParams()
    
    # Create and configure evaluator
    scanner = ReservoirPerformanceEvaluator(
        task=task_config['function'],
        param_name=param_name,
        param_range=param_range,
        param_grid=param_grid,
        result_keys=task_config['result_keys'],
        result_labels=task_config['result_labels'],
        reservoir_params=reservoir_params,
        extra_args=extra_args,
        reservoir_tag=reservoir_tag
    )
    
    # Log evaluation start
    print(f"Starting {task_config['description']}")
    summary = scanner.get_summary()
    print(f"Configuration: {summary['evaluation_mode']} mode")
    if summary['evaluation_mode'] == 'multi-parameter':
        print(f"Parameter grid: {param_grid}")
        print(f"Total combinations: {summary['total_combinations']}")
    else:
        print(f"Parameter: {param_name}, Range size: {len(param_range) if param_range else 0}")
    
    # Execute evaluation
    results = scanner.evaluate(save_dir=result_dir)
    
    print(f"Evaluation completed. Results saved with tag: {reservoir_tag}")
    return results

def get_available_tasks():
    """
    Get information about available evaluation tasks.
    
    Returns:
        dict: Task information including descriptions and result keys
    """
    return {task: {k: v for k, v in config.items() if k != 'function'} 
            for task, config in TASK_REGISTRY.items()}

def validate_task_config(task_type, **kwargs):
    """
    Validate configuration for a specific task type.
    
    Args:
        task_type (str): Task type to validate
        **kwargs: Configuration parameters to validate
    
    Returns:
        dict: Validation results and recommendations
    """
    task_key = task_type.upper()
    if task_key not in TASK_REGISTRY:
        return {'valid': False, 'error': f'Unknown task type: {task_type}'}
    
    task_config = TASK_REGISTRY[task_key]
    recommendations = []
    
    # Task-specific validation logic can be added here
    if task_key == 'MC':
        if 'signal_len' in kwargs and kwargs['signal_len'] < 200:
            recommendations.append('Consider signal_len >= 500 for reliable MC estimation')
    
    elif task_key in ['KRANDGR', 'KR_GR', 'KR&GR']:
        if 'threshold' in kwargs and kwargs['threshold'] > 0.5:
            recommendations.append('High threshold may result in low rank estimates')
    
    return {
        'valid': True,
        'task_description': task_config['description'],
        'expected_results': task_config['result_keys'],
        'recommendations': recommendations
    }