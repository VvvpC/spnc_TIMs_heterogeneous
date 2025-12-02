# -*- coding: utf-8 -*-
"""
NARMA10 evaluation task implementation.

This module provides the main evaluation function for NARMA10 nonlinear
system identification using reservoir computing systems.
"""

import numpy as np
from spnc import spnc_anisotropy
from core.reservoir import RunSpnc
from core.base_utils import MSE, NRMSE
import spnc_ml as ml

def evaluate_NARMA10(reservoir_params, Ntrain=2000, Ntest=1000, **kwargs):
    """
    Evaluate NARMA10 nonlinear system identification performance.
    
    This function uses the existing spnc_ml.spnc_narma10 implementation
    to evaluate the reservoir's capability for nonlinear processing.
    
    Args:
        reservoir_params (ReservoirParams): Reservoir configuration
        Ntrain (int): Number of training samples
        Ntest (int): Number of test samples
        **kwargs: Additional evaluation options
            seed_NARMA (int): Random seed for NARMA generation (default: 1234)
    
    Returns:
        dict: Results containing 'NRMSE', 'y_test', and 'pred' values
    
    Example:
        >>> params = ReservoirParams(Nvirt=50, beta_prime=30)
        >>> result = evaluate_NARMA10(params, Ntrain=3000, Ntest=1500)
        >>> print(f"NARMA10 NRMSE: {result['NRMSE']:.4f}")
    """
    seed_NARMA = kwargs.get('seed_NARMA', 1234)
    
    try:
        # Create SPNC system
        spn = spnc_anisotropy(
            reservoir_params.h,
            reservoir_params.theta_H,
            reservoir_params.k_s_0,
            reservoir_params.phi,
            reservoir_params.beta_prime,
            restart=True
        )
        
        transform = spn.gen_signal_slow_delayed_feedback
        
        # Use existing spnc_ml implementation for NARMA10
        (y_test, pred) = ml.spnc_narma10(
            Ntrain, Ntest,
            reservoir_params.Nvirt,
            reservoir_params.m0,
            reservoir_params.bias,
            transform,
            reservoir_params.params,
            seed_NARMA=seed_NARMA,
            fixed_mask=True,
            return_outputs=True
        )
        
        # Calculate NRMSE
        nrmse = NRMSE(pred, y_test)
        
        print(f"NARMA10 evaluation - Nvirt: {reservoir_params.Nvirt}, "
              f"NRMSE: {nrmse:.4f}")
        
        return {'NRMSE': nrmse, 'y_test': y_test, 'pred': pred}
        
    except Exception as e:
        print(f"Error in NARMA10 evaluation: {e}")
        # Return NaN to indicate failed evaluation
        return {'NRMSE': np.nan, 'y_test': None, 'pred': None}