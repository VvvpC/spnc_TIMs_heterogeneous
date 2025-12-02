# -*- coding: utf-8 -*-
"""
KR&GR evaluation task implementation.

This module provides the main evaluation function for Kernel Rank and
Generalization Rank assessment of reservoir computing systems.
"""

import numpy as np
from spnc import spnc_anisotropy
from core.reservoir import RunSpnc
from .signals import generate_kr_gr_input, preprocess_kr_gr_states, validate_kr_gr_inputs
from .processing import evaluate_kr_gr_ranks, calculate_computational_quality

def evaluate_KRandGR(reservoir_params, Nreadouts=50, Nwash=10, **kwargs):
    """
    Evaluate Kernel Rank and Generalization Rank of a reservoir system.
    
    This function implements the KR&GR evaluation protocol:
    1. Generate specialized input patterns (KR: diverse, GR: repeated)
    2. Run reservoir computation for each input pattern
    3. Collect reservoir states from all computations
    4. Analyze state matrices using SVD to determine ranks
    5. Calculate Computational Quality (CQ = KR - GR)
    
    Args:
        reservoir_params (ReservoirParams): Reservoir configuration
        Nreadouts (int): Number of readout points (overridden by reservoir_params.Nvirt)
        Nwash (int): Length of input sequences for KR evaluation
        **kwargs: Additional evaluation options
            threshold (float): Singular value threshold (default: 0.1)
            seed (int): Random seed (default: 1234)
            optimize_threshold (bool): Whether to optimize threshold (default: False)
    
    Returns:
        dict: Results containing 'KR', 'GR', and 'CQ' values
    
    Example:
        >>> params = ReservoirParams(Nvirt=30, beta_prime=20)
        >>> result = evaluate_KRandGR(params, Nwash=7)
        >>> print(f"KR: {result['KR']}, GR: {result['GR']}, CQ: {result['CQ']}")
    """
    # Use Nvirt from reservoir_params as the actual number of readouts
    Nreadouts = reservoir_params.Nvirt
    
    # Extract evaluation parameters
    threshold = kwargs.get('threshold', 0.1)
    seed = kwargs.get('seed', 1234)
    optimize_threshold = kwargs.get('optimize_threshold', False)
    
    try:
        # Generate KR&GR specific inputs
        inputs = generate_kr_gr_input(Nreadouts, Nwash, seed=seed)
        inputs = validate_kr_gr_inputs(inputs, Nreadouts)
        
        # Collect states from all input patterns
        outputs = []
        for input_row in inputs:
            input_row = input_row.reshape(-1, 1)
            
            # Create fresh SPNC instance for each computation
            spn = spnc_anisotropy(
                reservoir_params.h,
                reservoir_params.theta_H,
                reservoir_params.k_s_0,
                reservoir_params.phi,
                reservoir_params.beta_prime,
                restart=True
            )
            
            transform = spn.gen_signal_slow_delayed_feedback
            
            # Run reservoir computation
            output = RunSpnc(
                input_row, 1, 1,
                reservoir_params.Nvirt,
                reservoir_params.m0,
                transform,
                reservoir_params.params,
                fixed_mask=True,
                seed_mask=1234
            )
            
            outputs.append(output)
        
        # Stack all outputs into state tensor
        states = np.stack(outputs, axis=0)
        
        # Preprocess states for KR&GR analysis
        states, _, _ = preprocess_kr_gr_states(states, Nreadouts)
        
        # Optimize threshold if requested
        if optimize_threshold:
            from .processing import optimize_kr_gr_threshold
            opt_result = optimize_kr_gr_threshold(states, Nreadouts)
            threshold = opt_result['recommended_threshold']
            print(f"Optimized threshold: {threshold:.6f}")
        
        # Calculate KR and GR
        KR, GR = evaluate_kr_gr_ranks(states, Nreadouts, threshold=threshold)
        
        # Calculate Computational Quality
        CQ = calculate_computational_quality(KR, GR)
        
        print(f"KR&GR evaluation - Nvirt: {reservoir_params.Nvirt}, "
              f"threshold: {threshold:.3f}, KR: {KR}, GR: {GR}, CQ: {CQ}")
        
        return {'KR': KR, 'GR': GR, 'CQ': CQ}
        
    except Exception as e:
        print(f"Error in KR&GR evaluation: {e}")
        # Return NaN values to indicate failed evaluation
        return {'KR': np.nan, 'GR': np.nan, 'CQ': np.nan}