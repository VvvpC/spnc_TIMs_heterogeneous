# -*- coding: utf-8 -*-
"""
Processing functions optimized for Memory Capacity evaluation.

This module contains MC-specific implementations of mathematical operations,
optimized for the particular requirements of memory capacity assessment.
"""

import numpy as np
from core.base_utils import safe_numpy_convert, ensure_positive_definite

def ridge_regression_for_mc(states, target, l, bias=True):
    """
    Ridge regression implementation optimized for Memory Capacity tasks.
    
    This version is specifically tuned for MC evaluation:
    - Handles multiple target delays simultaneously
    - Optimized matrix operations for MC use case
    - Robust regularization for numerical stability
    
    Args:
        states (array-like): Reservoir states (N_samples x N_features)
        target (array-like): Target values (N_samples x N_delays)
        l (float): Regularization parameter (lambda)
        bias (bool): Whether to add bias term
    
    Returns:
        np.array: Regression weights for all delays
    """
    # Ensure numpy format
    states = safe_numpy_convert(states)
    target = safe_numpy_convert(target)
    
    if bias:
        # Add bias column
        bias_col = np.ones((len(states), 1))
        states = np.concatenate((bias_col, states), axis=1)
    
    # Optimized matrix computation for MC
    M1 = np.matmul(states.transpose(), target)
    M2 = np.matmul(states.transpose(), states)
    
    # Ensure numerical stability
    M2_reg = ensure_positive_definite(M2, l)
    
    # Solve linear system
    weights = np.matmul(np.linalg.pinv(M2_reg), M1)
    
    return weights

def preprocess_mc_states(states):
    """
    Preprocess reservoir states specifically for MC evaluation.
    
    Args:
        states (array-like): Raw reservoir states
    
    Returns:
        np.array: Preprocessed states optimized for MC
    """
    states = safe_numpy_convert(states)
    
    # MC-specific preprocessing could include:
    # - Normalization strategies
    # - Outlier handling  
    # - Dimensionality considerations
    
    return states

def postprocess_mc_results(mc_results):
    """
    Post-process MC results for final reporting.
    
    Args:
        mc_results (dict): Raw MC evaluation results
    
    Returns:
        dict: Processed results with additional metrics
    """
    processed = mc_results.copy()
    
    # Add derived metrics if needed
    if 'MC' in processed:
        processed['MC_normalized'] = min(processed['MC'] / 10.0, 1.0)  # Normalize to [0,1]
    
    return processed

def validate_mc_inputs(signal, states, splits, delays):
    """
    Validate inputs for Memory Capacity evaluation.
    
    Args:
        signal: Input signal
        states: Reservoir states
        splits: Data split ratios
        delays: Number of delays
    
    Returns:
        tuple: (validated_signal, validated_states)
    
    Raises:
        ValueError: If inputs are invalid
    """
    signal = safe_numpy_convert(signal)
    states = safe_numpy_convert(states)
    
    if len(signal) != len(states):
        raise ValueError(f"Signal length ({len(signal)}) != states length ({len(states)})")
    
    if len(signal) < delays * 2:
        raise ValueError(f"Signal too short ({len(signal)}) for delays ({delays})")
    
    if not (0 < splits[0] < splits[1] < 1):
        raise ValueError(f"Invalid splits {splits}. Must be 0 < splits[0] < splits[1] < 1")
    
    return signal, states