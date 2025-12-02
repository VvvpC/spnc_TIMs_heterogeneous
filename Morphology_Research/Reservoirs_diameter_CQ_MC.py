# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15:24:04 2025

@author: Chen

This script evaluates the MC and CQ performance of homogeneous superparamagnetic nanodot reservoirs 
with different sizes (beta_prime values) while maintaining the same input rate.

The key feature is using get_signal_slow_delayed_feedback_heteroRes_sameinput to ensure 
constant input_rate (theta_T*T) across different reservoir sizes.

"""

import numpy as np
from spnc import spnc_anisotropy

# Import all necessary functions from the formal framework
from formal_Parameter_Dynamics_Preformance import (
    generate_signal, linear_MC, gen_KR_GR_input, Evaluate_KR_GR, RunSpnc,
    ReservoirParams, ReservoirPerformanceEvaluator, run_evaluation
)

# ------------------------ Reservoir Size Parameters ----------------------------

class ReservoirSizeParams(ReservoirParams):
    """
    Extended ReservoirParams class for reservoir size evaluation with reference beta_prime
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add reference beta_prime for consistent input rate
        self.ref_beta_prime = kwargs.get('ref_beta_prime', 20)

    def print_params(self, verbose=True):
        if not verbose:
            print(f"ReservoirSizeParams(h={self.h}, beta_prime={self.beta_prime}, ref_beta_prime={self.ref_beta_prime}, Nvirt={self.Nvirt})")
        else:
            print(f"ReservoirSizeParams detailed info:")
            for attr in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'ref_beta_prime', 'Nvirt', 'm0', 'bias']:
                print(f"  {attr} = {getattr(self, attr)}")
            print("  params dictionary:")
            for k, v in self.params.items():
                print(f"    {k}: {v}")

# ------------------------ Size-Specific Task Functions ----------------------------

def evaluate_size_MC(reservoir_params, signal_len=550, **kwargs):
    """
    Evaluate Memory Capacity using heteroRes_sameinput transform to maintain constant input rate
    """
    signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))

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
    def transform_with_constant_rate(K_s, params, *args, **kwargs):
        return spn.gen_signal_slow_delayed_feedback_omegacons(
            K_s, params, reservoir_params.ref_beta_prime
        )
        # return spn.gen_signal_fast_delayed_feedback_omegacons(
        #     K_s, params, reservoir_params.ref_beta_prime
        # )

    Output = RunSpnc(
        signal,
        1,                 
        len(signal),       
        reservoir_params.Nvirt,
        reservoir_params.m0,
        transform_with_constant_rate,
        reservoir_params.params,
        fixed_mask=True,
        seed_mask=1234
    )

    MC = linear_MC(signal, Output, splits=[0.2, 0.6], delays=10)

    return {'MC': MC}

def evaluate_size_CQ(reservoir_params, Nreadouts=50, Nwash=10, **kwargs):
    """
    Evaluate Computational Quality (KR & GR) using heteroRes_sameinput transform
    """
    Nreadouts = reservoir_params.Nvirt

    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=1234)
    outputs = []
    
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        
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
        def transform_with_constant_rate(K_s, params, *args, **kwargs):
            return spn.gen_signal_slow_delayed_feedback_omegacons(
                K_s, params, reservoir_params.ref_beta_prime
            )
            # return spn.gen_signal_fast_delayed_feedback_omegacons(
            #     K_s, params, reservoir_params.ref_beta_prime
            # )
        
        output = RunSpnc(
            input_row, 1, 1, reservoir_params.Nvirt,
            reservoir_params.m0, transform_with_constant_rate, 
            reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
        outputs.append(output)
    
    States = np.stack(outputs, axis=0)
    States = States/np.amax(States)
    KR, GR = Evaluate_KR_GR(States, Nreadouts, threshold=0.001)
    CQ = KR - GR
    
    return {'KR': KR, 'GR': GR, 'CQ': CQ}

def evaluate_size_MC_CQ(reservoir_params, **kwargs):
    """
    Evaluate both MC and CQ for a given reservoir size
    """
    mc_result = evaluate_size_MC(reservoir_params, **kwargs)
    cq_result = evaluate_size_CQ(reservoir_params, **kwargs)
    
    return {
        'MC': mc_result['MC'],
        'KR': cq_result['KR'],
        'GR': cq_result['GR'],
        'CQ': cq_result['CQ']
    }

# ------------------------ Main Evaluation Function ----------------------------

def run_reservoir_size_evaluation(
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
    Run reservoir size evaluation for different beta_prime values
    
    Parameters:
    - task_type: 'MC', 'CQ', or 'MC_CQ'
    - beta_prime_range: array of beta_prime values to evaluate
    - reservoir_params: ReservoirSizeParams object
    - result_dir: directory to save results
    - plot: whether to show plots
    - verbose: whether to print detailed info
    - extra_args: additional arguments for task functions
    - filename_prefix: prefix for output files
    """
    
    # Map task types to evaluation functions
    task_map = {
        'MC': (evaluate_size_MC, ['MC'], ['Memory Capacity']),
        'CQ': (evaluate_size_CQ, ['KR', 'GR', 'CQ'], ['KR', 'GR', 'CQ']),
        'MC_CQ': (evaluate_size_MC_CQ, ['MC', 'KR', 'GR', 'CQ'], ['MC', 'KR', 'GR', 'CQ']),
        'MCCQ': (evaluate_size_MC_CQ, ['MC', 'KR', 'GR', 'CQ'], ['MC', 'KR', 'GR', 'CQ'])
    }
    
    if task_type.upper() not in task_map:
        raise ValueError(f"Unknown task_type: {task_type}. Available options: {list(task_map.keys())}")
    
    task, result_keys, result_labels = task_map[task_type.upper()]
    
    if reservoir_params is None:
        reservoir_params = ReservoirSizeParams()

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
        filename_prefix = f"reservoir_size_{task.__name__}"
    
    return evaluator.evaluate(
        save_dir=result_dir, 
        plot=plot, 
        verbose=verbose, 
        filename_prefix=filename_prefix
    )

# ------------------------ Beta Prime + Gamma Coupled Evaluation Functions ----------------------------

def calculate_gamma_from_beta_prime(beta_prime):
    """
    Calculate gamma value from beta_prime using the given equation:
    gamma = 9.66e-5*beta_prime^2 - 8.8e-3*beta_prime + 0.248 + 0.01121939974938757
    
    Parameters:
    - beta_prime: float or array-like, beta_prime values
    
    Returns:
    - gamma: float or array-like, calculated gamma values
    """
    # for alex's data
    # gamma = 9.66e-5 * beta_prime**2 - 8.8e-3 * beta_prime + 0.25306
    # for pareto-front value point
    gamma = 9.66e-5 * beta_prime**2 - 8.8e-3 * beta_prime + 0.22554
    return gamma

class ReservoirBetaGammaEvaluator:
    """
    Custom evaluator for beta_prime and gamma coupled evaluation
    """
    def __init__(self, task, beta_prime_range, result_keys, result_labels, 
                 reservoir_params=None, extra_args=None, 
                 use_gamma_calculation=True, gamma_range=None):
        """
        初始化评估器
        
        Parameters:
        -----------
        task : callable
            评估任务函数
        beta_prime_range : array-like
            beta_prime参数范围
        result_keys : list
            结果字典的键列表
        result_labels : list
            结果标签列表
        reservoir_params : ReservoirParams, optional
            储层参数对象
        extra_args : dict, optional
            额外参数
        use_gamma_calculation : bool, default=True
            是否使用calculate_gamma_from_beta_prime函数自动计算gamma
        gamma_range : array-like, optional
            手动指定的gamma范围（当use_gamma_calculation=False时必需）
        """
        self.task = task
        self.beta_prime_range = beta_prime_range
        self.result_keys = result_keys
        self.result_labels = result_labels
        self.reservoir_params = reservoir_params 
        self.extra_args = extra_args or {}
        self.use_gamma_calculation = use_gamma_calculation
        
        # 验证gamma_range
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
        
        # 根据开关决定如何获取gamma值
        if self.use_gamma_calculation:
            # 自动计算gamma值
            gamma_range = calculate_gamma_from_beta_prime(self.beta_prime_range)
            print(f"Using automatic gamma calculation")
        else:
            # 使用手动提供的gamma值
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

        print(f"Starting beta_prime + gamma coupled evaluation...")
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
            filename_prefix = f"beta_gamma_coupled_{self.task.__name__}"
        
        # Handle single value case
        if len(self.beta_prime_range) == 1:
            filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}_single_value.pkl"
        else:
            step = self.beta_prime_range[1] - self.beta_prime_range[0] if len(self.beta_prime_range) > 1 else 0
            filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}to{self.beta_prime_range[-1]}_step{step}.pkl"         
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
        elif len(self.result_keys) == 4:  # MC, KR, GR, CQ  
            ax1.plot(result_dict['beta_prime'], result_dict['MC'], 'o-', label='MC', linewidth=2, markersize=6)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(result_dict['beta_prime'], result_dict['CQ'], 'd--', label='CQ', color='red', linewidth=2, markersize=6)
            # ax1_twin.plot(result_dict['beta_prime'], result_dict['KR'], 's--', label='KR', color='orange', linewidth=2, markersize=6)
            # ax1_twin.plot(result_dict['beta_prime'], result_dict['GR'], '^:', label='GR', color='green', linewidth=2, markersize=6)
            ax1.set_ylabel('MC', fontsize=12)
            ax1_twin.set_ylabel('CQ', fontsize=12)
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
        elif len(self.result_keys) == 4:  # MC, KR, GR, CQ
            ax3.plot(result_dict['gamma'], result_dict['MC'], 'o-', label='MC', linewidth=2, markersize=6)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(result_dict['gamma'], result_dict['CQ'], 'd--', label='CQ', color='red', linewidth=2, markersize=6)
            #   ax3_twin.plot(result_dict['gamma'], result_dict['KR'], 's--', label='KR', color='orange', linewidth=2, markersize=6)
            # ax3_twin.plot(result_dict['gamma'], result_dict['GR'], '^:', label='GR', color='green', linewidth=2, markersize=6)
            ax3.set_ylabel('MC', fontsize=12)
            ax3_twin.set_ylabel('CQ', fontsize=12)
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        
        ax3.set_xlabel('Gamma', fontsize=12)
        ax3.set_title('Performance vs Gamma', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Plot 4: 3D scatter plot (Beta Prime vs Gamma vs MC)
        ax4 = axes[1, 1]
        if 'MC' in result_dict:
            scatter = ax4.scatter(result_dict['beta_prime'], result_dict['gamma'], 
                                c=result_dict['MC'], cmap='viridis', s=60)
            ax4.set_xlabel('Beta Prime', fontsize=12)
            ax4.set_ylabel('Gamma', fontsize=12)
            ax4.set_title('MC Performance Map', fontsize=14)
            plt.colorbar(scatter, ax=ax4, label='MC')
        
        fig.tight_layout()

        # Save plot
        if len(self.beta_prime_range) > 1:
            plot_filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}to{self.beta_prime_range[-1]}_step{self.beta_prime_range[1]-self.beta_prime_range[0]}.png"
        else:
            plot_filename = f"{filename_prefix}_evaluate_beta_prime_{self.beta_prime_range[0]}.png"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(save_dir, plot_filename)}")

        if show_plot:
            plt.show()
        plt.close(fig)

def run_reservoir_beta_gamma_evaluation(
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
    Run reservoir evaluation with coupled beta_prime and gamma parameters
    
    Parameters:
    -----------
    task_type : str
        'MC', 'CQ', or 'MC_CQ'
    beta_prime_range : array-like
        array of beta_prime values to evaluate
    reservoir_params : ReservoirSizeParams object
        储层参数对象
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
        是否使用自动gamma计算公式
    gamma_range : array-like, optional
        手动指定的gamma范围（当use_gamma_calculation=False时必需）
    
    Notes:
    ------
    当use_gamma_calculation=True时，gamma值自动通过以下公式计算:
    gamma = 9.66e-5*beta_prime^2 - 8.8e-3*beta_prime + 0.248
    
    当use_gamma_calculation=False时，必须提供gamma_range参数，
    且其长度必须与beta_prime_range相同。
    """
    
    # Map task types to evaluation functions
    task_map = {
        'MC': (evaluate_size_MC, ['MC'], ['Memory Capacity']),
        'CQ': (evaluate_size_CQ, ['KR', 'GR', 'CQ'], ['KR', 'GR', 'CQ']),
        'MC_CQ': (evaluate_size_MC_CQ, ['MC', 'KR', 'GR', 'CQ'], ['MC', 'KR', 'GR', 'CQ']),
        'MCCQ': (evaluate_size_MC_CQ, ['MC', 'KR', 'GR', 'CQ'], ['MC', 'KR', 'GR', 'CQ'])
    }
    
    if task_type.upper() not in task_map:
        raise ValueError(f"Unknown task_type: {task_type}. Available options: {list(task_map.keys())}")
    
    task, result_keys, result_labels = task_map[task_type.upper()]
    
    if reservoir_params is None:
        reservoir_params = ReservoirSizeParams()

    # Use the custom ReservoirBetaGammaEvaluator with new parameters
    evaluator = ReservoirBetaGammaEvaluator(
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
        method_suffix = "fitted_gamma" if use_gamma_calculation else "fixed_gamma"
        filename_prefix = f"diameterchange_{method_suffix}_{task.__name__}"
    
    return evaluator.evaluate(
        save_dir=result_dir, 
        plot=plot, 
        verbose=verbose, 
        filename_prefix=filename_prefix
    )

# ------------------------ Beta Gamma Heatmap Functions ----------------------------

def plot_beta_gamma_heatmap(reservoir_params=None, save_path=None, save_data=True):
    """
    绘制beta_prime和gamma的热力图
    
    Parameters:
    -----------
    reservoir_params : ReservoirSizeParams, optional
        储层参数对象，如果为None则使用默认参数
    save_path : str, optional
        图像保存路径，如果为None则显示图像
    save_data : bool, optional
        是否保存原始数据，默认为True
    """
    import matplotlib.pyplot as plt
    import tqdm
    
    # 参数网格设置
    beta_prime_range = np.linspace(25, 35, 10)
    gamma_range = np.linspace(0.05, 0.09, 10)
    
    # 初始化结果矩阵
    cq_matrix = np.zeros((10, 10))
    mc_matrix = np.zeros((10, 10))
    
    print(f"开始计算热力图数据")
    print(f"Beta_prime 范围: {beta_prime_range[0]:.1f} - {beta_prime_range[-1]:.1f}")
    print(f"Gamma 范围: {gamma_range[0]:.3f} - {gamma_range[-1]:.3f}")
    
    # 使用默认参数或提供的参数
    if reservoir_params is None:
        reservoir_params = ReservoirSizeParams(
            ref_beta_prime=30,
            h=0.4,
            Nvirt=200,
            m0=0.003,
            params={
                'theta': 0.3,
                'gamma': 0.05,
                'delay_feedback': 0,
                'Nvirt': 200,
            }
        )
    
    # 固定其他参数
    original_theta = reservoir_params.params.get('theta', 0.3)
    original_m0 = reservoir_params.m0
    
    # 遍历参数网格
    total_combinations = len(gamma_range) * len(beta_prime_range)
    current_idx = 0
    
    for i, gamma in enumerate(gamma_range):
        for j, beta_prime in enumerate(beta_prime_range):
            current_idx += 1
            try:
                # 更新参数
                reservoir_params.update_params(beta_prime=beta_prime)
                reservoir_params.params['gamma'] = gamma
                reservoir_params.params['theta'] = original_theta
                reservoir_params.m0 = original_m0
                
                # 评估MC和CQ
                result = evaluate_size_MC_CQ(reservoir_params, signal_len=550, seed=1234)
                
                MC = float(result.get("MC", 0.0))
                KR = float(result.get("KR", 0.0))
                GR = float(result.get("GR", 0.0))
                CQ = KR - GR
                
                cq_matrix[i, j] = CQ
                mc_matrix[i, j] = MC
                
                print(f"进度 {current_idx}/{total_combinations}: γ={gamma:.4f}, β'={beta_prime:.1f}, CQ={CQ:.3f}, MC={MC:.3f}")
                
            except Exception as e:
                print(f"计算失败 γ={gamma:.4f}, β'={beta_prime:.1f}: {e}")
                cq_matrix[i, j] = 0
                mc_matrix[i, j] = 0
    
    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CQ热力图
    im1 = ax1.imshow(cq_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('CQ热力图', fontsize=14)
    ax1.set_xlabel('beta_prime', fontsize=12)
    ax1.set_ylabel('gamma', fontsize=12)
    
    # 设置刻度标签
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels([f'{x:.1f}' for x in beta_prime_range])
    ax1.set_yticklabels([f'{x:.4f}' for x in gamma_range])
    
    plt.colorbar(im1, ax=ax1, label='CQ')
    
    # MC热力图
    im2 = ax2.imshow(mc_matrix, cmap='plasma', aspect='auto', origin='lower')
    ax2.set_title('MC热力图', fontsize=14)
    ax2.set_xlabel('beta_prime', fontsize=12)
    ax2.set_ylabel('gamma', fontsize=12)
    
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels([f'{x:.1f}' for x in beta_prime_range])
    ax2.set_yticklabels([f'{x:.4f}' for x in gamma_range])
    
    plt.colorbar(im2, ax=ax2, label='MC')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
    else:
        plt.show()
    
    # 保存原始数据
    result_data = {
        'cq_matrix': cq_matrix, 
        'mc_matrix': mc_matrix, 
        'beta_prime_range': beta_prime_range, 
        'gamma_range': gamma_range,
        'reservoir_params': {
            'ref_beta_prime': reservoir_params.ref_beta_prime,
            'h': reservoir_params.h,
            'Nvirt': reservoir_params.Nvirt,
            'm0': reservoir_params.m0,
            'params': reservoir_params.params.copy()
        }
    }
    
    if save_data:
        import pickle
        import os
        from datetime import datetime
        
        # 生成数据文件名
        if save_path:
            # 如果提供了图像路径，将数据保存在相同目录，文件名添加_data后缀
            base_name = os.path.splitext(save_path)[0]
            data_path = f"{base_name}_data.pkl"
        else:
            # 默认数据文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = f"beta_gamma_heatmap_data_{timestamp}.pkl"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else '.', exist_ok=True)
        
        # 保存数据
        with open(data_path, 'wb') as f:
            pickle.dump(result_data, f)
        print(f"热力图数据已保存到: {data_path}")
    
    return result_data

# ------------------------ Example Usage ----------------------------

# if __name__ == "__main__":
#     # Set up parameters
#     ref_beta_prime = 35.13826524755751
#     # Create reservoir parameters with reference beta_prime
#     reservoir_params = ReservoirSizeParams(
#         ref_beta_prime=ref_beta_prime,
#         h=0.4,
#         Nvirt=267,
#         m0=0.005288612874870094,
#         params={
#             'theta': 0.34142235979698393,
#             'gamma': 0.069274461903986,  # This will be overridden by the equation
#             'delay_feedback': 0,
#             'Nvirt':267,
#         }
#     )
    
#     beta_prime_range = np.arange(30, 40.5, 0.5)  # Range of beta_prime values to test

#     results_auto = run_reservoir_beta_gamma_evaluation(
#         task_type='MC_CQ',
#         beta_prime_range=beta_prime_range,
#         reservoir_params=reservoir_params,
#         result_dir="./results",
#         plot=True,
#         verbose=False,
#         use_gamma_calculation=True,
#         # gamma_range = [0.2 for _ in range(11)],
#         filename_prefix=None
#     )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="储层直径与CQ-MC性能评估")
    parser.add_argument("--heatmap", action="store_true", help="绘制beta_prime和gamma热力图")
    parser.add_argument("--save_path", type=str, default=None, help="热力图保存路径")
    parser.add_argument("--no_save_data", action="store_true", help="不保存原始数据")
    
    args = parser.parse_args()
    
    if args.heatmap:
        # 绘制热力图
        # 使用文件中现有的优化参数
        reservoir_params = ReservoirSizeParams(
            ref_beta_prime=30,
            h=0.4,
            Nvirt=200,
            m0=0.003,
            params={
                'theta': 0.3,
                'gamma': 0.076,
                'delay_feedback': 0,
                'Nvirt': 200,
            }
        )
        
        save_path = args.save_path if args.save_path else "beta_gamma_heatmap_2.png"
        save_data = not args.no_save_data  # 默认保存数据，除非指定--no_save_data
        result = plot_beta_gamma_heatmap(reservoir_params=reservoir_params, save_path=save_path, save_data=save_data)
        print("热力图绘制完成!")
    else:
        # 原始的beta-gamma耦合评估
        ref_beta_prime = 30
        # Create reservoir parameters with reference beta_prime
        reservoir_params = ReservoirSizeParams(
            ref_beta_prime=ref_beta_prime,
            h=0.4,
            Nvirt=200,
            m0=0.003,
            params={
                'theta': 0.3,
                'gamma': 0.0767,  # This will be overridden by the equation
                'delay_feedback': 0,
                'Nvirt':200,
            }
        )
        
        beta_prime_range = np.arange(25, 35.5, 0.5) # Range of beta_prime values to test

        results_auto = run_reservoir_beta_gamma_evaluation(
            task_type='MC_CQ',
            beta_prime_range=beta_prime_range,
            reservoir_params=reservoir_params,
            result_dir="./results",
            plot=True,
            verbose=False,
            use_gamma_calculation=False,
            gamma_range = [0.0767 for _ in range(21)],
            filename_prefix=None
        )