# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15:24:04 2025

@author: Chen

This script evaluates the NARMA10 and TI46 performance of homogeneous superparamagnetic nanodot reservoirs 
with different sizes (beta_prime values) and different gamma values.

The key feature is using get_signal_slow_delayed_feedback_heteroRes_sameinput to ensure 
constant input_rate (theta_T*T) across different reservoir sizes, while evaluating
computational tasks like NARMA10 and TI46 (spoken digits).

"""

import numpy as np
from spnc import spnc_anisotropy

# Import all necessary functions from the formal framework
from formal_Parameter_Dynamics_Preformance import (
    ReservoirParams, ReservoirPerformanceEvaluator, run_evaluation, RunSpnc
)

# Import ML tasks
import spnc_ml as ml

# ------------------------ Reservoir Size Parameters for Tasks ----------------------------

class ReservoirTaskParams(ReservoirParams):
    """
    Extended ReservoirParams class for reservoir size evaluation with task-specific parameters
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add reference beta_prime for consistent input rate
        self.ref_beta_prime = kwargs.get('ref_beta_prime', 20)
        # Add task-specific parameters
        self.Ntrain = kwargs.get('Ntrain', 2000)
        self.Ntest = kwargs.get('Ntest', 1000)
        self.speakers = kwargs.get('speakers', None)  # None means all speakers for TI46

    def print_params(self, verbose=True):
        if not verbose:
            print(f"ReservoirTaskParams(h={self.h}, beta_prime={self.beta_prime}, ref_beta_prime={self.ref_beta_prime}, Nvirt={self.Nvirt})")
        else:
            print(f"ReservoirTaskParams detailed info:")
            for attr in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'ref_beta_prime', 'Nvirt', 'm0', 'bias', 'Ntrain', 'Ntest', 'speakers']:
                print(f"  {attr} = {getattr(self, attr)}")
            print("  params dictionary:")
            for k, v in self.params.items():
                print(f"    {k}: {v}")

# ------------------------ Size-Specific Task Functions ----------------------------

def evaluate_size_NARMA10(reservoir_params, **kwargs):
    """
    Evaluate NARMA10 performance using heteroRes_sameinput transform to maintain constant input rate
    """
    # Create spnc with current beta_prime
    spn = spnc_anisotropy(
        reservoir_params.h,
        reservoir_params.theta_H,
        reservoir_params.k_s_0,
        reservoir_params.phi,
        reservoir_params.beta_prime,
        restart=True
    )

    # Create transform function that maintains constant input rate
    def transform_with_constant_rate(signal, params, *args, **kwargs):
        return spn.gen_signal_slow_delayed_feedback_omegacons(
            signal, params, reservoir_params.ref_beta_prime 
        )

    # Use the NARMA10 function from spnc_ml
    Ntrain = kwargs.get('Ntrain', reservoir_params.Ntrain)
    Ntest = kwargs.get('Ntest', reservoir_params.Ntest)
    
    NRMSE = ml.spnc_narma10(
        Ntrain, Ntest, reservoir_params.Nvirt,
        reservoir_params.m0, reservoir_params.bias,
        transform_with_constant_rate, reservoir_params.params,
        seed_NARMA=kwargs.get('seed_NARMA', 1234), 
        fixed_mask=kwargs.get('fixed_mask', True),
        seed_mask=kwargs.get('seed_mask', 1234),
        return_NRMSE=True
    )

    return {'NRMSE': NRMSE}

def evaluate_size_TI46(reservoir_params, **kwargs):
    """
    Evaluate TI46 (spoken digits) performance using heteroRes_sameinput transform
    """
    # Create spnc with current beta_prime
    spn = spnc_anisotropy(
        reservoir_params.h,
        reservoir_params.theta_H,
        reservoir_params.k_s_0,
        reservoir_params.phi,
        reservoir_params.beta_prime,
        restart=True
    )

    # Create transform function that maintains constant input rate
    def transform_with_constant_rate(signal, params, *args, **kwargs):
        return spn.gen_signal_slow_delayed_feedback_omegacons(
            signal, params, reservoir_params.ref_beta_prime, 
        )

    # Use the TI46 function from spnc_ml
    speakers = kwargs.get('speakers', reservoir_params.speakers)
    
    accuracy = ml.spnc_TI46(
        speakers, reservoir_params.Nvirt,
        reservoir_params.m0, reservoir_params.bias,
        transform_with_constant_rate, reservoir_params.params,
        fixed_mask=kwargs.get('fixed_mask', True),
        seed_mask=kwargs.get('seed_mask', 1234),
    )

    return {'Accuracy': accuracy}

def evaluate_size_NARMA10_TI46(reservoir_params, **kwargs):
    """
    Evaluate both NARMA10 and TI46 for a given reservoir size
    """
    narma10_result = evaluate_size_NARMA10(reservoir_params, **kwargs)
    ti46_result = evaluate_size_TI46(reservoir_params, **kwargs)
    
    return {
        'NRMSE': narma10_result['NRMSE'],
        'Accuracy': ti46_result['Accuracy']
    }

# ------------------------ Main Evaluation Function ----------------------------

def run_reservoir_size_tasks_evaluation(
    task_type,
    beta_prime_range,
    reservoir_params=None,
    result_dir="./results",
    plot=True,
    verbose=False,
    extra_args=None,
    filename_prefix=None
):
    """
    Run reservoir size evaluation for different beta_prime values with NARMA10 and TI46 tasks
    
    Parameters:
    - task_type: 'NARMA10', 'TI46', or 'NARMA10_TI46'
    - beta_prime_range: array of beta_prime values to evaluate
    - reservoir_params: ReservoirTaskParams object
    - result_dir: directory to save results
    - plot: whether to show plots
    - verbose: whether to print detailed info
    - extra_args: additional arguments for task functions
    - filename_prefix: prefix for output files
    """
    
    # Map task types to evaluation functions
    task_map = {
        'NARMA10': (evaluate_size_NARMA10, ['NRMSE'], ['NRMSE']),
        'TI46': (evaluate_size_TI46, ['Accuracy'], ['Accuracy']),
        'NARMA10_TI46': (evaluate_size_NARMA10_TI46, ['NRMSE', 'Accuracy'], ['NRMSE', 'Accuracy'])
    }
    
    if task_type.upper() not in task_map:
        raise ValueError(f"Unknown task_type: {task_type}. Available options: {list(task_map.keys())}")
    
    task, result_keys, result_labels = task_map[task_type.upper()]
    
    if reservoir_params is None:
        reservoir_params = ReservoirTaskParams()

    # Use the existing ReservoirPerformanceEvaluator framework
    evaluator = ReservoirPerformanceEvaluator(
        task=task,
        param_name='beta_prime',
        param_range=beta_prime_range,
        result_keys=result_keys,
        result_labels=result_labels,
        reservoir_params=reservoir_params,
        extra_args=extra_args or {}
    )
    
    # Set default filename prefix if not provided
    if filename_prefix is None:
        filename_prefix = f"reservoir_size_tasks_{task.__name__}"
    
    return evaluator.evaluate(
        save_dir=result_dir, 
        plot=plot, 
        verbose=verbose, 
        filename_prefix=filename_prefix
    )

# ------------------------ Beta Prime + Gamma Coupled Evaluation for Tasks ----------------------------

def calculate_gamma_from_beta_prime(beta_prime):
    """
    Calculate gamma value from beta_prime using the given equation:
    gamma = 9.66e-5*beta_prime^2 - 8.8e-3*beta_prime + 0.248
    
    Parameters:
    - beta_prime: float or array-like, beta_prime values
    
    Returns:
    - gamma: float or array-like, calculated gamma values
    """
    gamma = 9.66e-5 * beta_prime**2 - 8.8e-3 * beta_prime + 0.22554
    return gamma

class ReservoirTaskBetaGammaEvaluator:
    """
    Custom evaluator for beta_prime and gamma coupled evaluation with tasks
    """
    def __init__(self, task, beta_prime_range, result_keys, result_labels, 
                 reservoir_params=None, extra_args=None, 
                 use_gamma_calculation=True, gamma_range=None):
        self.task = task
        self.beta_prime_range = beta_prime_range
        self.result_keys = result_keys
        self.result_labels = result_labels
        self.reservoir_params = reservoir_params 
        self.extra_args = extra_args or {}
        self.use_gamma_calculation = use_gamma_calculation
        
        # Validate gamma_range
        if not use_gamma_calculation:
            if gamma_range is None:
                raise ValueError("gamma_range must be provided when use_gamma_calculation=False")
            if len(gamma_range) != len(beta_prime_range):
                raise ValueError(f"gamma_range length ({len(gamma_range)}) must match beta_prime_range length ({len(beta_prime_range)})")
            self.gamma_range = gamma_range
        else:
            self.gamma_range = None

    def evaluate(self, save_dir='./results', plot=False, verbose=False, filename_prefix=None):
        import os
        import pickle
        import matplotlib.pyplot as plt
        import tqdm
        
        # Determine gamma values based on setting
        if self.use_gamma_calculation:
            gamma_range = calculate_gamma_from_beta_prime(self.beta_prime_range)
            print(f"Using automatic gamma calculation: γ = 9.66e-5β² - 8.8e-3β + 0.248")
        else:
            gamma_range = self.gamma_range
            print(f"Using manually provided gamma values")
        
        # Initialize result dictionary
        result_dict = {
            'beta_prime': self.beta_prime_range,
            'gamma': gamma_range,
            'gamma_calculation_method': 'automatic' if self.use_gamma_calculation else 'manual'
        }
        for key in self.result_keys:
            result_dict[key] = []

        print(f"Starting beta_prime + gamma coupled evaluation for tasks...")
        print(f"Beta_prime range: {self.beta_prime_range}")
        print(f"Gamma range: [{', '.join([f'{x:.8f}' for x in gamma_range])}]")

        for i, (beta_val, gamma_val) in enumerate(tqdm.tqdm(zip(self.beta_prime_range, gamma_range), 
                                                            total=len(self.beta_prime_range),
                                                            desc="Evaluating coupled parameters")):
            
            # Update both beta_prime and gamma
            self.reservoir_params.update_params(beta_prime=beta_val)
            self.reservoir_params.params['gamma'] = gamma_val
            
            if verbose:
                print(f"\n--- Evaluation {i+1}/{len(self.beta_prime_range)} ---")
                print(f"Beta_prime: {beta_val:.2f}, Gamma: {gamma_val:.6f}")
                self.reservoir_params.print_params(verbose=True)

            try:
                task_result = self.task(self.reservoir_params, **self.extra_args)
                for key in self.result_keys:
                    result_dict[key].append(task_result[key])

            except Exception as e:
                print(f"Error evaluating beta_prime={beta_val}, gamma={gamma_val}: {e}")
                for key in self.result_keys:
                    result_dict[key].append(np.nan)

        # Save results
        if filename_prefix is None:
            filename_prefix = f"beta_gamma_coupled_tasks_{self.task.__name__}"
        
        filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}to{self.beta_prime_range[-1]}_step{self.beta_prime_range[1]-self.beta_prime_range[0]:.1f}.pkl"         
        save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(result_dict, f)
        print(f"Results saved to {save_path}")

        # Plot results 
        self._plot_results(result_dict, save_dir, filename_prefix, plot)

        return result_dict

    def _plot_results(self, result_dict, save_dir, filename_prefix, show_plot):
        import matplotlib.pyplot as plt
        import os
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Results vs Beta Prime
        ax1 = axes[0, 0]
        if len(self.result_keys) == 1:
            ax1.plot(result_dict['beta_prime'], result_dict[self.result_keys[0]], 'o-', linewidth=2, markersize=6)
            ax1.set_ylabel(self.result_labels[0], fontsize=12)
        elif len(self.result_keys) == 2:  # NRMSE and Accuracy
            ax1.plot(result_dict['beta_prime'], result_dict['NRMSE'], 'o-', label='NRMSE', linewidth=2, markersize=6)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(result_dict['beta_prime'], result_dict['Accuracy'], 's--', label='Accuracy', color='orange', linewidth=2, markersize=6)
            ax1.set_ylabel('NRMSE', fontsize=12)
            ax1_twin.set_ylabel('Accuracy', fontsize=12)
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
        
        ax1.set_xlabel('Beta Prime', fontsize=12)
        ax1.set_title('Performance vs Beta Prime', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gamma vs Beta Prime
        ax2 = axes[0, 1]
        ax2.plot(result_dict['beta_prime'], result_dict['gamma'], 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Beta Prime', fontsize=12)
        ax2.set_ylabel('Gamma', fontsize=12)
        ax2.set_title('Gamma vs Beta Prime\n(γ = 9.66e-5β² - 8.8e-3β + 0.248)', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Results vs Gamma
        ax3 = axes[1, 0]
        if len(self.result_keys) == 1:
            ax3.plot(result_dict['gamma'], result_dict[self.result_keys[0]], 'o-', linewidth=2, markersize=6)
            ax3.set_ylabel(self.result_labels[0], fontsize=12)
        elif len(self.result_keys) == 2:
            ax3.plot(result_dict['gamma'], result_dict['NRMSE'], 'o-', label='NRMSE', linewidth=2, markersize=6)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(result_dict['gamma'], result_dict['Accuracy'], 's--', label='Accuracy', color='orange', linewidth=2, markersize=6)
            ax3.set_ylabel('NRMSE', fontsize=12)
            ax3_twin.set_ylabel('Accuracy', fontsize=12)
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        
        ax3.set_xlabel('Gamma', fontsize=12)
        ax3.set_title('Performance vs Gamma', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Plot 4: 3D scatter plot (Beta Prime vs Gamma vs primary metric)
        ax4 = axes[1, 1]
        if 'NRMSE' in result_dict:
            scatter = ax4.scatter(result_dict['beta_prime'], result_dict['gamma'], 
                                c=result_dict['NRMSE'], cmap='viridis', s=60)
            ax4.set_xlabel('Beta Prime', fontsize=12)
            ax4.set_ylabel('Gamma', fontsize=12)
            ax4.set_title('NRMSE Performance Map', fontsize=14)
            plt.colorbar(scatter, ax=ax4, label='NRMSE')
        elif 'Accuracy' in result_dict:
            scatter = ax4.scatter(result_dict['beta_prime'], result_dict['gamma'], 
                                c=result_dict['Accuracy'], cmap='plasma', s=60)
            ax4.set_xlabel('Beta Prime', fontsize=12)
            ax4.set_ylabel('Gamma', fontsize=12)
            ax4.set_title('Accuracy Performance Map', fontsize=14)
            plt.colorbar(scatter, ax=ax4, label='Accuracy')
        
        fig.tight_layout()

        # Save plot
        plot_filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}to{self.beta_prime_range[-1]}_step{self.beta_prime_range[1]-self.beta_prime_range[0]:.1f}.png"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(save_dir, plot_filename)}")

        if show_plot:
            plt.show()
        plt.close(fig)

def run_reservoir_beta_gamma_tasks_evaluation(
    task_type,
    beta_prime_range,
    reservoir_params=None,
    result_dir="./results",
    plot=True,
    verbose=False,
    extra_args=None,
    filename_prefix=None,
    use_gamma_calculation=True,
    gamma_range=None
):
    """
    Run reservoir evaluation with coupled beta_prime and gamma parameters for tasks
    
    Parameters:
    -----------
    task_type : str
        'NARMA10', 'TI46', or 'NARMA10_TI46'
    beta_prime_range : array-like
        array of beta_prime values to evaluate
    reservoir_params : ReservoirTaskParams object
        reservoir parameters object
    result_dir : str
        directory to save results
    plot : bool
        whether to show plots
    verbose : bool
        whether to print detailed info
    extra_args : dict
        additional arguments for task functions
    filename_prefix : str
        prefix for output files
    use_gamma_calculation : bool, default=True
        whether to use automatic gamma calculation formula
    gamma_range : array-like, optional
        manual gamma range (required when use_gamma_calculation=False)
    """
    
    # Map task types to evaluation functions
    task_map = {
        'NARMA10': (evaluate_size_NARMA10, ['NRMSE'], ['NRMSE']),
        'TI46': (evaluate_size_TI46, ['Accuracy'], ['Accuracy']),
        'NARMA10_TI46': (evaluate_size_NARMA10_TI46, ['NRMSE', 'Accuracy'], ['NRMSE', 'Accuracy'])
    }
    
    if task_type.upper() not in task_map:
        raise ValueError(f"Unknown task_type: {task_type}. Available options: {list(task_map.keys())}")
    
    task, result_keys, result_labels = task_map[task_type.upper()]
    
    if reservoir_params is None:
        reservoir_params = ReservoirTaskParams()

    # Use the custom ReservoirTaskBetaGammaEvaluator
    evaluator = ReservoirTaskBetaGammaEvaluator(
        task=task,
        beta_prime_range=beta_prime_range,
        result_keys=result_keys,
        result_labels=result_labels,
        reservoir_params=reservoir_params,
        extra_args=extra_args or {},
        use_gamma_calculation=use_gamma_calculation,
        gamma_range=gamma_range
    )
    
    # Set default filename prefix if not provided
    if filename_prefix is None:
        method_suffix = "auto_gamma" if use_gamma_calculation else "manual_gamma"
        filename_prefix = f"beta_gamma_coupled_tasks_{method_suffix}_{task.__name__}"
    
    return evaluator.evaluate(
        save_dir=result_dir, 
        plot=plot, 
        verbose=verbose, 
        filename_prefix=filename_prefix
    )

# ------------------------ Example Usage ----------------------------

if __name__ == "__main__":
    # Set up parameters
    ref_beta_prime = 29.14406097255966
    # Create reservoir parameters with reference beta_prime
    reservoir_params = ReservoirTaskParams(
        ref_beta_prime=ref_beta_prime,
        h=0.4,
        Nvirt=200,
        m0=0.004703581408469578,
        Ntrain=2000,  # Reduced for faster testing
        Ntest=1000,    # Reduced for faster testing
        speakers=['f1', 'f2', 'f3', 'f4', 'f5'],  # Use all speakers
        params={
            'theta': 0.13798577326972078,
            'gamma': 0.05110574322049721,  # This will be overridden by the equation
            'delay_feedback': 0,
            'Nvirt': 200,
        }
    )
    
    beta_prime_range = np.arange(25, 35.5, 0.5)   # Small range for testing
    
    # # Example 1: Evaluate NARMA10 task only
    # print("=== Evaluating NARMA10 Task ===")
    # results_narma10 = run_reservoir_beta_gamma_tasks_evaluation(
    #     task_type='NARMA10',
    #     beta_prime_range=beta_prime_range,
    #     reservoir_params=reservoir_params,
    #     result_dir="./results",
    #     plot=False,
    #     verbose=False,
    #     use_gamma_calculation=True,
    #     filename_prefix="narma10_tasks_example"
    # )
    
    # # Example 2: Evaluate TI46 task only
    # print("\n=== Evaluating TI46 Task ===")
    # results_ti46 = run_reservoir_beta_gamma_tasks_evaluation(
    #     task_type='TI46',
    #     beta_prime_range=beta_prime_range,
    #     reservoir_params=reservoir_params,
    #     result_dir="./results",
    #     plot=False,
    #     verbose=False,
    #     use_gamma_calculation=True,
    #     filename_prefix="ti46_tasks_example"
    # )
    
    # Example 3: Evaluate both tasks
    print("\n=== Evaluating Both NARMA10 and TI46 Tasks ===")
    results_both = run_reservoir_beta_gamma_tasks_evaluation(
        task_type='NARMA10_TI46',
        beta_prime_range=beta_prime_range,
        reservoir_params=reservoir_params,
        result_dir="./results",
        plot=True,
        verbose=False,
        use_gamma_calculation=True,
        # gamma_range=[0.05110574322049721 for _ in range(21)],
        filename_prefix="auto_gamma1"
    )