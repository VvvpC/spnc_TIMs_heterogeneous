# -*- coding: utf-8 -*-
"""
TI46 speech recognition evaluation task implementation.

This module provides the main evaluation function for TI46 digit
recognition using reservoir computing systems.
"""

import numpy as np
from spnc import spnc_anisotropy
from core.reservoir import RunSpnc
import spnc_ml as ml

def evaluate_Ti46(reservoir_params, **kwargs):
    """
    Evaluate TI46 speech recognition performance.
    
    This function uses the existing spnc_ml.spnc_TI46 implementation
    to evaluate the reservoir's speech recognition capability.
    
    Args:
        reservoir_params (ReservoirParams): Reservoir configuration
        **kwargs: Additional evaluation options
            nvirt_ti46 (int): Override Nvirt for TI46 task (default: use reservoir_params.Nvirt)
            speakers (list): List of speakers to evaluate (default: ['f1','f2','f3','f4','f5'])
    
    Returns:
        dict: Results containing 'acc' (accuracy) value
    
    Example:
        >>> params = ReservoirParams(Nvirt=40, beta_prime=25)
        >>> result = evaluate_Ti46(params)
        >>> print(f"TI46 Accuracy: {result['acc']:.3f}")
    """
    # Get TI46-specific parameters
    nvirt_override = kwargs.get('nvirt_ti46', reservoir_params.Nvirt)
    speakers = kwargs.get('speakers', ['f1','f2','f3','f4','f5'])
    
    try:
        # Create temporary parameters for TI46 task
        temp_params = reservoir_params.params.copy()
        temp_params['Nvirt'] = nvirt_override
        
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
        
        # Use existing spnc_ml implementation for TI46
        acc = ml.spnc_TI46(
            speakers,
            nvirt_override,
            reservoir_params.m0,
            reservoir_params.bias,
            transform,
            temp_params
        )
        
        print(f"TI46 evaluation - Nvirt: {nvirt_override}, "
              f"Accuracy: {acc:.3f}")
        
        return {'acc': acc}
        
    except Exception as e:
        print(f"Error in TI46 evaluation: {e}")
        # Return NaN to indicate failed evaluation
        return {'acc': np.nan}