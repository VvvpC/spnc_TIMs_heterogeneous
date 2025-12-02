# -*- coding: utf-8 -*-
"""
Signal generation and processing functions for Memory Capacity evaluation.

This module contains all signal-related operations specific to MC tasks:
- Random signal generation with washout
- Linear memory capacity calculation
- Delay-based target signal construction
"""

import numpy as np
from core.base_utils import safe_numpy_convert
from .processing import ridge_regression_for_mc

def generate_mc_signal(length, washout=50, seed=1234):
    """
    Generate i.i.d. random signal sequence for Memory Capacity evaluation.
    
    The signal is uniformly distributed in [-1, 1] range, which is optimal
    for linear memory capacity assessment.
    
    Args:
        length (int): Total signal length including washout
        washout (int): Number of initial samples to discard
        seed (int): Random seed for reproducibility
    
    Returns:
        np.array: Washed signal sequence shaped as (length-washout, 1)
    """
    if seed is not None:
        np.random.seed(seed)
        
    signal_sequence = np.random.uniform(-1, 1, size=length)
    washed_signal = signal_sequence[washout:]
    
    return washed_signal.reshape(-1, 1)

def linear_memory_capacity(signal, states, splits=[0.2, 0.8], delays=50):
    """
    Calculate linear memory capacity using delay-based reconstruction.
    
    This function implements the standard MC evaluation protocol:
    1. Create delayed target signals
    2. Split data into train/test sets
    3. Optimize Ridge regression hyperparameter
    4. Calculate MC score for each delay
    5. Sum valid MC scores (>0.1 threshold)
    
    Args:
        signal (array-like): Input signal sequence
        states (array-like): Reservoir state outputs
        splits (list): Data split ratios [washout_end, train_end] 
        delays (int): Maximum delay for memory assessment
    
    Returns:
        float: Total linear memory capacity
    """
    # Ensure flat input signal
    signal = np.asarray(signal).flatten()
    
    # Generate delayed target matrix
    shift = _create_delay_matrix(signal, delays)
    
    # Split data according to specified ratios
    wash, Ytrain, Ytest = np.split(shift, [
        int(len(signal) * splits[0]), 
        int(len(signal) * splits[1])
    ])
    wash, Xtrain, Xtest = np.split(states, [
        int(len(signal) * splits[0]), 
        int(len(signal) * splits[1])
    ])
    
    # Hyperparameter optimization for best MC
    best_mc = 0
    gammas = np.logspace(-10, 0, 11)
    
    for gamma in gammas:
        # Train Ridge regression with current gamma
        weights = ridge_regression_for_mc(Xtrain, Ytrain, gamma, bias=False)
        
        # Predict on test set
        prediction = np.matmul(Xtest, weights)
        
        # Calculate MC score for all delays
        mc = _calculate_mc_score(prediction, Ytest, delays)
        
        if mc > best_mc:
            best_mc = mc
    
    return best_mc

def _create_delay_matrix(signal, delays):
    """
    Create matrix of delayed signal versions for MC evaluation.
    
    Args:
        signal (np.array): Input signal
        delays (int): Maximum delay
    
    Returns:
        np.array: Matrix where column k contains signal delayed by k steps
    """
    shift = np.zeros((len(signal), delays))
    for i in range(len(signal) - delays):
        i += delays
        shift[i, :] = signal[i-delays:i]
    return shift

def _calculate_mc_score(prediction, target, delays):
    """
    Calculate Memory Capacity score from predictions and targets.
    
    MC_k = cov(pred_k, target_k)^2 / (var(pred_k) * var(target_k))
    Only scores > 0.1 are considered significant.
    
    Args:
        prediction (np.array): Predicted values for all delays
        target (np.array): Target values for all delays  
        delays (int): Number of delays to evaluate
    
    Returns:
        float: Total memory capacity (sum of valid MC_k scores)
    """
    mc_k = np.zeros(delays)
    
    for k in range(delays):
        pred = prediction[:, k]
        targ = target[:, k]
        
        # Calculate covariance matrix
        M = pred, targ
        coVarM = np.cov(M)
        coVar = coVarM[0, 1]  # Cross-covariance
        
        # Calculate variances
        outVar = np.var(pred)
        targVar = np.var(targ)
        totVar = outVar * targVar
        
        # MC score with significance threshold
        if totVar > 1e-12 and coVar**2/totVar > 0.1:
            mc_k[k] = coVar**2/totVar
    
    # Account for floating point errors
    mc_k[mc_k > 1] = 1
    
    return sum(mc_k)