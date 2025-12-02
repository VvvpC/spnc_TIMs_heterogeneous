# -*- coding: utf-8 -*-
"""
Memory Capacity evaluation task implementation.

This module provides the main evaluation function for linear memory capacity
assessment of reservoir computing systems using superparamagnetic nanodots.
"""

from spnc import spnc_anisotropy
from core.reservoir import RunSpnc
from .signals import generate_mc_signal, linear_memory_capacity
from .processing import validate_mc_inputs

def evaluate_MC(reservoir_params, signal_len=550, **kwargs):
    """
    Evaluate linear memory capacity of a reservoir computing system.
    
    This function implements the standard MC evaluation protocol:
    1. Generate random input signal
    2. Configure SPNC system with given parameters
    3. Run reservoir computation
    4. Calculate linear memory capacity using delay reconstruction
    
    Args:
        reservoir_params (ReservoirParams): Reservoir parameter configuration
        signal_len (int): Length of input signal (including washout)
        **kwargs: Additional evaluation options
            seed (int): Random seed for reproducibility (default: 1234)
            splits (list): Data split ratios (default: [0.2, 0.6])
            delays (int): Maximum delay for MC evaluation (default: 10)
            washout (int): Washout period (default: 50)
    
    Returns:
        dict: Evaluation results with key 'MC' containing memory capacity value
    
    Example:
        >>> params = ReservoirParams(beta_prime=50, Nvirt=30)
        >>> result = evaluate_MC(params, signal_len=1000, delays=15)
        >>> print(f"Memory Capacity: {result['MC']:.3f}")
    """
    # Extract evaluation parameters
    seed = kwargs.get('seed', 1234)
    splits = kwargs.get('splits', [0.2, 0.6])
    delays = kwargs.get('delays', 10)
    washout = kwargs.get('washout', 50)
    
    try:
        # Generate MC-specific input signal
        signal = generate_mc_signal(signal_len, washout=washout, seed=seed)
        
        # Create SPNC system
        spn = spnc_anisotropy(
            reservoir_params.h,
            reservoir_params.theta_H,
            reservoir_params.k_s_0,
            reservoir_params.phi,
            reservoir_params.beta_prime,
            restart=True
        )
        
        # Get transformation function
        transform = spn.gen_signal_slow_delayed_feedback
        
        # Run reservoir computation
        output = RunSpnc(
            signal, 1, 1,
            reservoir_params.Nvirt,
            reservoir_params.m0,
            transform,
            reservoir_params.params,
            fixed_mask=True,
            seed_mask=1234
        )
        
        # Validate inputs before MC calculation
        signal_flat = signal.flatten()
        signal_flat, output = validate_mc_inputs(signal_flat, output, splits, delays)
        
        # Calculate linear memory capacity
        mc = linear_memory_capacity(signal_flat, output, splits=splits, delays=delays)
        
        # Log evaluation details
        print(f"MC evaluation - m0: {reservoir_params.m0:.6f}, "
              f"gamma: {reservoir_params.params['gamma']:.6f}, "
              f"MC: {mc:.3f}")
        
        return {'MC': mc}
        
    except Exception as e:
        print(f"Error in MC evaluation: {e}")
        # Return NaN to indicate failed evaluation
        import numpy as np
        return {'MC': np.nan}