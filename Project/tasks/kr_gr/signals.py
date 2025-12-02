# -*- coding: utf-8 -*-
"""
Signal generation for KR&GR evaluation tasks.

This module generates specialized input patterns for evaluating
Kernel Rank and Generalization Rank of reservoir systems.
"""

import numpy as np

def generate_kr_gr_input(nreadouts, nwash=10, seed=1234):
    """
    Generate input sequences for KR&GR evaluation.
    
    Creates two types of inputs:
    - KR inputs: Random sequences for each readout (tests kernel diversity)
    - GR inputs: Repeated pattern across readouts (tests generalization)
    
    The input structure is: [KR_inputs | GR_inputs]
    where KR_inputs has nwash columns and GR_inputs has 10 columns.
    
    Args:
        nreadouts (int): Number of readout points (typically equals Nvirt)
        nwash (int): Length of KR input sequence for each readout
        seed (int): Random seed for reproducibility
    
    Returns:
        np.array: Combined input matrix (nreadouts x (nwash + 10))
    """
    np.random.seed(seed)
    
    # Generate KR inputs: unique random sequence for each readout
    kr_inputs = np.random.ranf((nreadouts, nwash))
    
    # Generate GR inputs: same pattern repeated for all readouts
    gr_pattern = np.random.ranf(10)
    gr_inputs = np.tile(gr_pattern, (nreadouts, 1))
    
    # Combine KR and GR inputs
    all_inputs = np.concatenate((kr_inputs, gr_inputs), axis=1)
    
    return all_inputs

def preprocess_kr_gr_states(states, nreadouts):
    """
    Preprocess reservoir states for KR&GR analysis.
    
    Extracts and normalizes states corresponding to KR and GR evaluations.
    
    Args:
        states (np.array): Raw reservoir states (nreadouts x timesteps x features)
        nreadouts (int): Number of readouts
    
    Returns:
        tuple: (normalized_states, gr_states, kr_states)
    """
    # Extract GR states (last timestep) and KR states (specific timestep)
    gr_states = states[:, -1, :]   # States at last timestep for GR
    kr_states = states[:, -11, :]  # States at timestep -11 for KR
    
    # Normalize all states
    max_val = np.amax(np.abs(states))
    if max_val > 0:
        states_normalized = states / max_val
    else:
        states_normalized = states
    
    return states_normalized, gr_states, kr_states

def validate_kr_gr_inputs(inputs, expected_nreadouts):
    """
    Validate KR&GR input format and dimensions.
    
    Args:
        inputs (np.array): Input matrix to validate
        expected_nreadouts (int): Expected number of readouts
    
    Returns:
        np.array: Validated inputs
    
    Raises:
        ValueError: If inputs have incorrect format
    """
    inputs = np.asarray(inputs)
    
    if inputs.ndim != 2:
        raise ValueError(f"Inputs must be 2D, got {inputs.ndim}D")
    
    if inputs.shape[0] != expected_nreadouts:
        raise ValueError(f"Expected {expected_nreadouts} readouts, got {inputs.shape[0]}")
    
    if inputs.shape[1] < 10:  # Minimum for GR inputs
        raise ValueError(f"Input sequence too short: {inputs.shape[1]} < 10")
    
    return inputs