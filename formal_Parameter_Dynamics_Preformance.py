# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15:24:04 2025

@author: Chen

Thic script contains a framework for evaluating the MC, KRandGR and computational performance of a superparamagnetic nanodot system (spn) with varying parameters.

"""

# import necessary libraries
import os
import torch 
import torch.nn as nn
from spnc import spnc_anisotropy
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import pickle
import spnc_ml as ml
import itertools


from pathlib import Path


CANDIDATES = [
    
    Path(r"C:\Users\tom\Desktop\Repository"),
    Path(r"C:\Users\Chen\Desktop\Repository"),
    Path(r"/Users/vvvp./Desktop"),
]
searchpaths = [p for p in CANDIDATES if p.exists()]

#tuple of repos
repos = ('machine_learning_library',)

from deterministic_mask import fixed_seed_mask, max_sequences_mask
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from sklearn.metrics import classification_report

# General functions

# -----------------MC----------------------------

def generate_signal(I,washout = 50,seed=1234):
    '''
    Generate a i.i.d. signal sequence with a [-1,1] range
    '''
    if seed is not None:
        np.random.seed(seed)
        
    signal_sequence = np.random.uniform(-1,1,size = I)

    washed_signal = signal_sequence[washout:]

    # Convert to 2D array
    washed_signal = washed_signal.reshape(-1,1)

    # print(np.shape(washed_signal))

    return washed_signal

def RidgeRegression(states, target, l, bias=True):
    # Ensure numpy
    if torch.is_tensor(states):
        states = states.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if bias==True:
        # Add bias to states
        bias = np.ones((len(states), 1))
        states = np.concatenate((bias, states), axis=1)
    # Setup matrices from inputs
    M1 = np.matmul(states.transpose(), target) 
    M2 = np.matmul(states.transpose(), states)
    # Perform ridge regression
    weights = np.matmul(np.linalg.pinv(M2+l*np.identity(len(M2))), M1)
    return weights

def linear_MC(signal, states, splits=[0.2, 0.8], delays=50):
    # ensure flat input signal
    signal = np.asarray(signal).flatten()
    # generate target signal from delayed input signal
    shift = np.zeros((len(signal), delays))
    for i in range(len(signal)-delays):
        i += delays
        shift[i, :] = signal[i-delays:i]
    # split data
    wash, Ytrain, Ytest = np.split(shift, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    wash, Xtrain, Xtest = np.split(states, [int(len(signal)*splits[0]), int(len(signal)*splits[1])])
    # sweep over range of hyperparameters gamma to find optimal MC
    bestMC = 0
    gammas = np.logspace(-10, 0, 11)
    for gamma in gammas:
        # Calculate weights
        weights = RidgeRegression(Xtrain, Ytrain, gamma, bias=False)
        # Predict test states
        prediction = np.matmul(Xtest, weights)
        # Loop over all delays k and evaluate MC_k
        MC_k = np.zeros(delays)
        for k in range(delays):
            # Take prediction and target for each delay
            pred = prediction[:, k]
            targ = Ytest[:, k]
            # Set up matrix to calculate covariance
            M = pred, targ
            # Calculate covariance
            coVarM = np.cov(M)
            # Take cov(xy) 
            coVar = coVarM[0,1]
            # Measure the variance of the signals
            outVar = np.var(pred)
            targVar = np.var(targ)
            # Calculate the total variance of the raw target and the specific
            # target
            totVar = outVar*targVar
            # If the covariance coefficient is greater than 0.1, treat as better
            # than random guessing and add to MC_k outputs
            if coVar**2/totVar > 0.1:
                MC_k[k] = coVar**2/totVar
        # Account for floating point errors in MC
        MC_k[MC_k>1] = 1
        # Sum memory capacity over all delays
        MC = sum(MC_k)
        # If best reported MC, save data
        if MC > bestMC:
            bestMC = MC
    return bestMC

# 整合成一个
# ------------------------ KRandGR ----------------------------
'''
23/06/25 Chen


Generate input with more equal figures for KRandGR, last 7  columns are GR inputs, the rest are KR inputs.

Here Nwash = 7 for KR, rest of 7 columns are GR


'''
def gen_KR_GR_input(Nreadouts, Nwash=10, seed=1234):
    # set seed
    np.random.seed(seed)
    # generate KR inputs
    KR_inputs = np.random.ranf((Nreadouts, Nwash))
    GR_inputs = np.tile(np.random.ranf((10)), (Nreadouts,1))
    all_inputs = np.concatenate((KR_inputs, GR_inputs), axis=1)
    # 打印all_inputs的前10个元素
    return all_inputs


def Evaluate_KR_GR(states, Nreadouts, threshold=0.1):
    GR_states = states[:,-1,:]
    '''
    Change the last 7 columns to GR states, the rest are KR states
    '''
    KR_states = states[:,-11,:]
    uGR, sGR, vGR = np.linalg.svd(GR_states)
    uKR, sKR, vKR = np.linalg.svd(KR_states)
    KR = 0
    GR = 0
    for i in range(Nreadouts):
        if sGR[i]>threshold:
            GR += 1
        if sKR[i]>threshold:
            KR += 1
    return KR, GR

# ------------------------ Reservoir ----------------------------
def RunSpnc(signal,Nin,Nout,Nvirt,m0,transform, params,**kwargs):
    '''
    Run a reservoir computer with the signal sequence
    '''
    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask==True:
        # print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            # print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            # print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    
    # Run
    S,_ = snr.transform(signal,params)
    
    return S

# ------------------------ Evaluation tasks ----------------------------
# ##########
# MC task function
# ##########
def evaluate_MC(reservoir_params, signal_len = 550, **kwargs):

    signal = generate_signal(signal_len, seed=kwargs.get('seed', 1234))
    # 打印signal的前10个元素
    spn = spnc_anisotropy(
        reservoir_params.h,
        reservoir_params.theta_H,
        reservoir_params.k_s_0,
        reservoir_params.phi,
        reservoir_params.beta_prime,
        restart=True
    )

    transform = spn.gen_signal_slow_delayed_feedback

    Output = RunSpnc(
        signal,
        1,                 
        1,       
        reservoir_params.Nvirt,
        reservoir_params.m0,
        transform,
        reservoir_params.params,
        fixed_mask=True,
        seed_mask=1234
    )
    MC = linear_MC(signal, Output, splits=[0.2,0.6], delays=10)
    print(f"res_m0: {reservoir_params.m0}, res_gamma: {reservoir_params.params['gamma']}, MC: {MC}")

    return {'MC': MC}

# ##########
# KRandGR task function
# ##########
def evaluate_KRandGR(reservoir_params, Nreadouts=50, Nwash=10, **kwargs):
    
    Nreadouts= reservoir_params.Nvirt
    inputs = gen_KR_GR_input(Nreadouts, Nwash, seed=1234)   # <--- 用Nreadouts
    outputs = []
    for input_row in inputs:
        input_row = input_row.reshape(-1, 1)
        spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                              reservoir_params.k_s_0, reservoir_params.phi,
                              reservoir_params.beta_prime, restart=True)
        transforms = spn.gen_signal_slow_delayed_feedback
        output = RunSpnc(input_row, 1, 1, reservoir_params.Nvirt,
                         reservoir_params.m0, transforms, reservoir_params.params, fixed_mask=True, seed_mask=1234)
        outputs.append(output)
    States = np.stack(outputs, axis=0)
    # rescale the stage by divide the maximum
    Normalized_States = States/np.amax(States)

    # rescale the stage to the range of [-1,1]
    # States_min = np.amin(States)
    # States_max = np.amax(States)
    # Normalized_States = 2 * (States - States_min) / (States_max - States_min) - 1
    # rescale the stage to the range of [0,1]
    # States_min = np.amin(States)
    # States_max = np.amax(States)
    # States = (States - States_min) / (States_max - States_min)
    # without rescaling
    # Normalized_States = States

    if kwargs.get('threshold') is not None:
        threshold = kwargs.get('threshold')
    else:
        threshold = 0.001
    KR, GR = Evaluate_KR_GR(Normalized_States, Nreadouts, threshold=threshold) 
    
    CQ = KR - GR 
    return {'KR': KR, 'GR': GR, 'CQ': CQ}


# ##########
# NARMA10 task function
# ##########
def MSE(pred, desired):
    return np.mean(np.square(np.subtract(pred, desired)))

def NRMSE(pred, y_test, spacer=0.001):
    return np.sqrt(MSE(pred, y_test) / np.var(y_test)) 


def evaluate_NARMA10(reservoir_params, Ntrain=2000, Ntest=1000, **kwargs):
    spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                          reservoir_params.k_s_0, reservoir_params.phi,
                          reservoir_params.beta_prime, restart=True)
    transform = spn.gen_signal_slow_delayed_feedback
    (y_test, pred) = ml.spnc_narma10(Ntrain, Ntest, reservoir_params.Nvirt,
                            reservoir_params.m0, reservoir_params.bias,
                            transform, reservoir_params.params,
                            seed_NARMA=1234, fixed_mask=True, return_outputs=True)
    nrmse = NRMSE(pred, y_test)

    return {'NRMSE': nrmse, 'y_test': y_test, 'pred': pred}

# ##########
# TI46 task function
# ##########

def evaluate_Ti46(reservoir_params, **kwargs):
    # 获取TI46专用的Nvirt，如果未指定则使用默认值
    nvirt_override = kwargs.get('nvirt_ti46', reservoir_params.Nvirt)
    
    # 创建临时参数副本用于TI46任务
    import copy
    temp_params = copy.deepcopy(reservoir_params)
    temp_params.Nvirt = nvirt_override
    temp_params.params['Nvirt'] = nvirt_override
    
    # 打印下temp_params的详细信息让我来确认是否override
    # from pprint import pprint
    # print("temp_params.__dict__:")
    # pprint(temp_params.__dict__)

    spn = spnc_anisotropy(reservoir_params.h, reservoir_params.theta_H,
                          reservoir_params.k_s_0, reservoir_params.phi,
                          reservoir_params.beta_prime, restart=True)
    transform = spn.gen_signal_slow_delayed_feedback
    speakers = ['f1','f2','f3','f4','f5']
    acc = ml.spnc_TI46(speakers, nvirt_override, reservoir_params.m0, reservoir_params.bias, transform, temp_params.params)
    return {'acc': acc}
    
# ------------------------ Reservoir Parameters Dictionary --------------------------

class  ReservoirParams:
    def __init__(self, **kwargs):
            # Reservoir parameters 
            self.h = 0.4
            self.theta_H = 90
            self.k_s_0 = 0
            self.phi = 45
            self.beta_prime = 20

            # Network parameters 
            self.Nvirt = 30
            self.m0 = 0.007586422893538462
            self.bias = True
            self.Nwarmup = 0
            self.verbose_repr = False

            self.params = {
                'theta': 0.5540233436467944,
                'gamma': 0.13738441393289658,
                'delay_feedback': 0,
                'Nvirt': self.Nvirt,
                'length_warmup': self.Nwarmup,
                'warmup_sample': self.Nwarmup * self.Nvirt,
                'voltage_noise': False,
                'seed_voltage_noise': 1234,
                'delta_V': 0.1,
                'johnson_noise': False,
                'seed_johnson_noise': 1234,
                'mean_johnson_noise': 0.0000,
                'std_johnson_noise': 0.00001,
                'thermal_noise': False,
                'seed_thermal_noise': 1234,
                'lambda_ou': 1.0,
                'sigma_ou': 0.1
        }

            for key in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 'Nvirt', 'm0', 'bias', 'Nwarmup']:
                if key in kwargs:
                    setattr(self, key, kwargs[key])

            
            if 'params' in kwargs and isinstance(kwargs['params'], dict):
                self.params.update(kwargs['params'])

    
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.params:
                self.params[key] = value
            if not hasattr(self, key) and key not in self.params:
                raise AttributeError(f"ReservoirParams has no attribute or param key '{key}'")
            
    def print_params(self, verbose=False):
        if not verbose:
            print(f"ReservoirParams(h={self.h}, beta_prime={self.beta_prime}, Nvirt={self.Nvirt})")
        else:
            print(f"ReservoirParams detailed info:")
            print(f"  h = {self.h}")
            print(f"  theta_H = {self.theta_H}")
            print(f"  k_s_0 = {self.k_s_0}")
            print(f"  phi = {self.phi}")
            print(f"  beta_prime = {self.beta_prime}")
            print(f"  Nvirt = {self.Nvirt}")
            print(f"  m0 = {self.m0}")
            print(f"  bias = {self.bias}")
            print("  params dictionary:")
            for k, v in self.params.items():
                print(f"    {k}: {v}")

# ##########
# A General Framwork for evaluating the MC, KRandGR and computational performance of a superparamagnetic nanodot system (spn) with varying parameters.
# ##########

class ReservoirPerformanceEvaluator:
    def __init__(self, task, param_name=None, param_range=None, param_grid=None, result_keys=None, result_labels=None, reservoir_params=None, extra_args=None, reservoir_tag='default'):
        self.task = task
        self.result_keys = result_keys
        self.result_labels = result_labels
        self.reservoir_params = reservoir_params 
        self.extra_args = extra_args or {}
        self.reservoir_tag = reservoir_tag
        
        # Determine if single-parameter or multi-parameter scanning
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
        """Generate all parameter combinations for multi-parameter grid search"""
        if not self.is_multi_param:
            return None
        
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries for easier handling
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(self.param_names, combo))
            param_combinations.append(param_dict)
        
        return param_combinations

    def evaluate(self, save_dir='./Res_Tasks_Results', verbose = False):
        
        # Initialize result dictionary
        if self.is_multi_param:
            result_dict = {'param_combinations': []}
            # Add each parameter name as a key for easy access
            for param_name in self.param_names:
                result_dict[param_name] = []
            # Prepare iteration over parameter combinations
            param_iterator = self.param_combinations
            total_iterations = len(self.param_combinations)
        else:
            result_dict = {self.param_name: self.param_range}
            param_iterator = self.param_range
            total_iterations = len(self.param_range)

        for key in self.result_keys:
            result_dict[key] = []

        # Main evaluation loop
        for i, param_config in enumerate(tqdm.tqdm(param_iterator, total=total_iterations)):
            if self.is_multi_param:
                # Update all parameters for multi-parameter case
                self.reservoir_params.update_params(**param_config)
                result_dict['param_combinations'].append(param_config.copy())
                # Also store individual parameter values for easy access
                for param_name, param_value in param_config.items():
                    result_dict[param_name].append(param_value)
                
                # Print current parameter values for verification
                print(f"[{i+1}/{total_iterations}] Config: {param_config}")
                print(f"  → reservoir_params.m0 = {self.reservoir_params.m0}")
                print(f"  → reservoir_params.Nvirt = {self.reservoir_params.Nvirt}")
                print(f"  → reservoir_params.beta_prime = {self.reservoir_params.beta_prime}")
                print(f"  → reservoir_params.params = {self.reservoir_params.params}")
                
                if verbose:
                    print(f"Evaluating combination {i+1}/{total_iterations}: {param_config}")
                    self.reservoir_params.print_params(verbose=True)
            else:
                # Update single parameter for backward compatibility
                self.reservoir_params.update_params(**{self.param_name: param_config})
                print(f"[{i+1}/{total_iterations}] Single param {self.param_name}={param_config}")
                print(f"  → reservoir_params.{self.param_name} = {getattr(self.reservoir_params, self.param_name, 'NOT_FOUND')}")
                
                if verbose:
                    print(f"Evaluating {self.param_name}={param_config}")
                    self.reservoir_params.print_params(verbose=True)

            try:
                task_result = self.task(self.reservoir_params, **self.extra_args)
                for key in self.result_keys:
                    result_dict[key].append(task_result[key])

            except Exception as e:
                if self.is_multi_param:
                    print(f"Error evaluating combination {param_config}: {e}")
                else:
                    print(f"Error evaluating {self.param_name}={param_config}: {e}")
                for key in self.result_keys:
                    result_dict[key].append(np.nan)

        # combine the result_dict with the reservoir_tag
        root_path = os.path.join(save_dir, f"Reservoir_{self.reservoir_tag}")

        os.makedirs(root_path, exist_ok=True)

        save_path = os.path.join(root_path, "results.pkl")

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                saved_results = pickle.load(f)
        else:
            saved_results = {'reservoir_tag': self.reservoir_tag, 'runs':[]}

        run_entry = {
            'task': getattr(self.task, '__name__', 'task'),
            'is_multi_param': self.is_multi_param,
            'results': result_dict
        }
        
        # Add parameter information based on scanning type
        if self.is_multi_param:
            run_entry['param_grid'] = self.param_grid
            run_entry['param_names'] = self.param_names
            run_entry['total_combinations'] = len(self.param_combinations)
        else:
            run_entry['param_name'] = self.param_name
            run_entry['param_range'] = self.param_range
        saved_results['runs'].append(run_entry)

        with open(save_path, 'wb') as f:
            pickle.dump(saved_results, f)

        print(f"Results saved to {save_path}")

        return result_dict
        
    

# ##########
# run the evaluation
# ##########

def run_evaluation(
    task_type,
    param_name=None,
    param_range=None,
    param_grid=None,
    reservoir_params=None,
    result_dir="./Res_Tasks_Results",   
    extra_args=None,
    reservoir_tag='default'
):

    if task_type.upper() == 'MC':
        task = evaluate_MC
        result_keys = ['MC']
        result_labels = ['Memory Capacity']
    elif task_type.upper() in ['KRANDGR', 'KR_GR', 'KR&GR']:
        task = evaluate_KRandGR
        result_keys = ['KR','GR','CQ']
        result_labels = ['KR','GR','CQ']
    elif task_type.upper() == 'NARMA10':
        task = evaluate_NARMA10
        result_keys = ['NRMSE', 'y_test', 'pred']
        result_labels = ['NRMSE', 'Desired', 'Predicted']
    elif task_type.upper() == 'TI46':
        task = evaluate_Ti46
        result_keys = ['acc']
        result_labels = ['TI46 Accuracy']
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    scanner = ReservoirPerformanceEvaluator(
        task=task,
        param_name=param_name,
        param_range=param_range,
        param_grid=param_grid,
        result_keys=result_keys,
        result_labels=result_labels,
        reservoir_params=reservoir_params or ReservoirParams(),
        extra_args=extra_args,
        reservoir_tag=reservoir_tag
    )
    return scanner.evaluate(save_dir=result_dir)

# Usage Examples:
# 
# ## Single Parameter Scanning (Backward Compatible)
# result = run_evaluation(
#     task_type='MC',
#     param_name='beta_prime',
#     param_range=[10, 20, 30, 50],
#     reservoir_params=ReservoirParams(),
#     reservoir_tag='single_param_test'
# )
#
# Multi-Parameter Grid Scanning (New Feature)
if __name__ == "__main__":
    reservoir_params = ReservoirParams(
        beta_prime=50,
        Nvirt=200,
        m0=0.008,
        params={'theta': 0.2, 'gamma': 0.1, 'Nvirt': 200}
    )

    all_results = {}
    task_types = ['MC', 'KRandGR', 'NARMA10', 'TI46']

    m0_range = np.linspace(0.03,0.18, 10)
    gamma_range = np.linspace(0.045, 0.1, 10)
    for task in task_types:
        print(f"\n>>> Running task: {task}")
        # Reset reservoir_params to initial state before each task
        reservoir_params.update_params(m0=0.008, **{'theta': 0.2, 'gamma': 0.1})
        result = run_evaluation(
            task_type=task,
            param_grid={'m0': m0_range, 'gamma': gamma_range},
            reservoir_params=reservoir_params,
            extra_args={'nvirt_ti46': 150},
            reservoir_tag='Res_m00.03-0.18_gamma0.045-0.1_max'
        )
        all_results[task] = result

# # load the results
# with open('./Res_Tasks_Results/Reservoir_Res_beta30-40_gamma0.05-0.06/results.pkl', 'rb') as f:
#     results = pickle.load(f)

#     MC_results = []
#     params = None


#     for run in results['runs']:
#         # Check if it's multi-parameter or single-parameter run
#         is_multi_param = run.get('is_multi_param', False)
        
#         if run['task'] == 'evaluate_MC':
#             if is_multi_param:
#                 # For multi-parameter, results are stored as lists with param_combinations
#                 MC_results.extend(run['results']['MC'])
#             else:
#                 # For single-parameter, append the entire result list
#                 MC_results.append(run['results']['MC'])
#     if results['runs'] and results['runs'][0].get('is_multi_param', False):
#         params = results['runs'][0]['results'].get('param_combinations', [])

#     print(MC_results[1])    
#     print(params[1])



# 
# ## Mixed Parameter Scanning
# scanner = ReservoirPerformanceEvaluator(
#     task=evaluate_MC,
#     param_name='beta_prime',
#     param_range=[50],
#     param_grid={'h': [0.2, 0.4], 'Nvirt': [20, 30]},  # This will override param_name/param_range
#     result_keys=['MC'],
#     result_labels=['Memory Capacity'],
#     reservoir_params=ReservoirParams(),
#     reservoir_tag='mixed_test'
# )

# if __name__ == "__main__":
#     # create a new ReservoirParams object
#     reservoir_params = ReservoirParams(
#         beta_prime=50,
#         Nvirt=10,
#         m0=0.008,
#         params={'theta': 0.3, 'gamma': 0.1}
#     )


#     # set the task types
#     task_types = ['MC', 'KRandGR', 'NARMA10', 'TI46']

#     all_results = {}

#     for task in task_types:
#         print(f"\n>>> Running task: {task}")
#         result = run_evaluation(
#             task_type=task,
#             param_name='beta_prime',
#             param_range=[50],   # 这里只有一个参数点
#             reservoir_params=reservoir_params,
#             result_dir="./Res_Tasks_Results",
#             reservoir_tag='Res_beta50'
#         )
#         all_results[task] = result

#     print("\n四个任务全部完成")


# # load the results
# with open('./Res_Tasks_Results/Reservoir_Res_beta50/results.pkl', 'rb') as f:
#     results = pickle.load(f)

# nrmse = []
# for run in results['runs']:
#     if run['task'] == 'evaluate_NARMA10':
#         nrmse.append(run['results']['NRMSE'])

# print(nrmse[0])



