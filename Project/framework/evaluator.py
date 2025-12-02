# -*- coding: utf-8 -*-
"""
Performance evaluation framework for reservoir computing systems.

This module provides the ReservoirPerformanceEvaluator class which handles
parameter scanning, evaluation orchestration, and result management.
"""

import os
import pickle
import itertools
import numpy as np
import tqdm

class ReservoirPerformanceEvaluator:
    """
    Comprehensive evaluation framework for reservoir computing performance.
    
    This class supports both single-parameter and multi-parameter grid search
    evaluation with automatic result saving and error handling.
    """
    
    def __init__(self, task, param_name=None, param_range=None, param_grid=None,
                 result_keys=None, result_labels=None, reservoir_params=None,
                 extra_args=None, reservoir_tag='default'):
        """
        Initialize the performance evaluator.
        
        Args:
            task (callable): Evaluation function to execute
            param_name (str): Single parameter name for scanning (optional)
            param_range (list): Values to scan for single parameter (optional)
            param_grid (dict): Multi-parameter grid {param_name: [values]} (optional)
            result_keys (list): Keys expected in task results
            result_labels (list): Human-readable labels for results
            reservoir_params (ReservoirParams): Base reservoir configuration
            extra_args (dict): Additional arguments passed to task function
            reservoir_tag (str): Tag for organizing saved results
        """
        self.task = task
        self.result_keys = result_keys or []
        self.result_labels = result_labels or []
        self.reservoir_params = reservoir_params
        self.extra_args = extra_args or {}
        self.reservoir_tag = reservoir_tag
        
        # Determine evaluation mode: single vs multi-parameter
        if param_grid is not None:
            self.is_multi_param = True
            self.param_grid = param_grid
            self.param_names = list(param_grid.keys())
            self.param_combinations = self._generate_param_combinations()
        elif param_name is not None and param_range is not None:
            self.is_multi_param = False
            self.param_name = param_name
            self.param_range = param_range
            self.param_names = [param_name]
        else:
            raise ValueError("Either (param_name, param_range) or param_grid must be provided")
    
    def _generate_param_combinations(self):
        """
        Generate all parameter combinations for multi-parameter grid search.
        
        Returns:
            list: List of parameter dictionaries for each combination
        """
        if not self.is_multi_param:
            return None
        
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(self.param_names, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def evaluate(self, save_dir='./Res_Tasks_Results', verbose=False):
        """
        Execute the evaluation process with parameter scanning.
        
        Args:
            save_dir (str): Directory to save results
            verbose (bool): Enable verbose logging
        
        Returns:
            dict: Complete evaluation results
        """
        # Initialize result storage
        if self.is_multi_param:
            result_dict = {'param_combinations': []}
            # Add individual parameter keys for easy access
            for param_name in self.param_names:
                result_dict[param_name] = []
            param_iterator = self.param_combinations
            total_iterations = len(self.param_combinations)
        else:
            result_dict = {self.param_name: self.param_range}
            param_iterator = self.param_range
            total_iterations = len(self.param_range)
        
        # Initialize result value storage
        for key in self.result_keys:
            result_dict[key] = []
        
        # Main evaluation loop with progress tracking
        for i, param_config in enumerate(tqdm.tqdm(param_iterator, total=total_iterations)):
            
            if self.is_multi_param:
                # Multi-parameter configuration
                self.reservoir_params.update_params(**param_config)
                result_dict['param_combinations'].append(param_config.copy())
                
                # Store individual parameter values
                for param_name, param_value in param_config.items():
                    result_dict[param_name].append(param_value)
                
                # Progress logging
                print(f"[{i+1}/{total_iterations}] Config: {param_config}")
                self._log_current_params(verbose)
                
            else:
                # Single-parameter configuration
                self.reservoir_params.update_params(**{self.param_name: param_config})
                print(f"[{i+1}/{total_iterations}] {self.param_name}={param_config}")
                
                if verbose:
                    print(f"Evaluating {self.param_name}={param_config}")
                    self.reservoir_params.print_params(verbose=True)
            
            # Execute task evaluation with error handling
            try:
                task_result = self.task(self.reservoir_params, **self.extra_args)
                
                # Extract and store results
                for key in self.result_keys:
                    if key in task_result:
                        result_dict[key].append(task_result[key])
                    else:
                        print(f"Warning: Expected key '{key}' not found in task result")
                        result_dict[key].append(np.nan)
            
            except Exception as e:
                # Error handling: record NaN for failed evaluations
                error_context = param_config if self.is_multi_param else f"{self.param_name}={param_config}"
                print(f"Error evaluating {error_context}: {e}")
                
                for key in self.result_keys:
                    result_dict[key].append(np.nan)
        
        # Save results
        self._save_results(result_dict, save_dir)
        
        return result_dict
    
    def _log_current_params(self, verbose):
        """Log current reservoir parameter state."""
        print(f"  → reservoir_params.m0 = {self.reservoir_params.m0}")
        print(f"  → reservoir_params.Nvirt = {self.reservoir_params.Nvirt}")
        print(f"  → reservoir_params.beta_prime = {self.reservoir_params.beta_prime}")
        
        if verbose:
            self.reservoir_params.print_params(verbose=True)
    
    def _save_results(self, result_dict, save_dir):
        """
        Save evaluation results to pickle file with metadata.
        
        Args:
            result_dict (dict): Results to save
            save_dir (str): Base directory for saving
        """
        # Create save directory structure
        root_path = os.path.join(save_dir, f"Reservoir_{self.reservoir_tag}")
        os.makedirs(root_path, exist_ok=True)
        save_path = os.path.join(root_path, "results.pkl")
        
        # Load existing results or create new structure
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                saved_results = pickle.load(f)
        else:
            saved_results = {'reservoir_tag': self.reservoir_tag, 'runs': []}
        
        # Create run entry with metadata
        run_entry = {
            'task': getattr(self.task, '__name__', str(self.task)),
            'is_multi_param': self.is_multi_param,
            'results': result_dict
        }
        
        # Add parameter configuration metadata
        if self.is_multi_param:
            run_entry['param_grid'] = self.param_grid
            run_entry['param_names'] = self.param_names
            run_entry['total_combinations'] = len(self.param_combinations)
        else:
            run_entry['param_name'] = self.param_name
            run_entry['param_range'] = self.param_range
        
        # Append to saved results
        saved_results['runs'].append(run_entry)
        
        # Save to disk
        with open(save_path, 'wb') as f:
            pickle.dump(saved_results, f)
        
        print(f"Results saved to {save_path}")
    
    def get_summary(self):
        """
        Get a summary of the evaluation configuration.
        
        Returns:
            dict: Configuration summary
        """
        summary = {
            'task_name': getattr(self.task, '__name__', str(self.task)),
            'evaluation_mode': 'multi-parameter' if self.is_multi_param else 'single-parameter',
            'reservoir_tag': self.reservoir_tag,
            'result_keys': self.result_keys
        }
        
        if self.is_multi_param:
            summary['param_grid'] = self.param_grid
            summary['total_combinations'] = len(self.param_combinations)
        else:
            summary['param_name'] = self.param_name
            summary['param_range_size'] = len(self.param_range)
        
        return summary