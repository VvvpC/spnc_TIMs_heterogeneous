# -*- coding: utf-8 -*-
"""
Processing functions for KR&GR evaluation.

This module implements the core algorithms for computing Kernel Rank
and Generalization Rank from reservoir states.
"""

import numpy as np
from core.base_utils import safe_numpy_convert

def evaluate_kr_gr_ranks(states, nreadouts, threshold=0.001):
    """
    Calculate Kernel Rank and Generalization Rank from reservoir states.
    
    Uses Singular Value Decomposition (SVD) to analyze the rank structure
    of KR and GR state matrices. Ranks are determined by counting singular
    values above the specified threshold.
    
    Args:
        states (np.array): Reservoir states (nreadouts x timesteps x features)
        nreadouts (int): Number of readout points
        threshold (float): Minimum singular value threshold for rank counting
    
    Returns:
        tuple: (KR, GR) - Kernel Rank and Generalization Rank values
    """
    states = safe_numpy_convert(states)
    
    # Extract GR and KR specific states
    gr_states = states[:, -1, :]   # Last timestep for GR evaluation
    kr_states = states[:, -11, :]  # Timestep -11 for KR evaluation
    
    # Compute SVD for both state matrices
    _, sGR, _ = np.linalg.svd(gr_states)
    _, sKR, _ = np.linalg.svd(kr_states)
    
    # Count singular values above threshold
    KR = np.sum(sKR > threshold)
    GR = np.sum(sGR > threshold)
    
    return KR, GR

def calculate_computational_quality(KR, GR):
    """
    Calculate Computational Quality (CQ) from KR and GR values.
    
    CQ = KR - GR represents the balance between kernel diversity
    and generalization capability.
    
    Args:
        KR (int): Kernel Rank
        GR (int): Generalization Rank
    
    Returns:
        int: Computational Quality score
    """
    return KR - GR

def analyze_rank_distribution(singular_values, threshold=0.001):
    """
    Analyze the distribution of singular values for rank assessment.
    
    Args:
        singular_values (np.array): Sorted singular values from SVD
        threshold (float): Threshold for significant singular values
    
    Returns:
        dict: Analysis results including rank, ratio, and spectrum info
    """
    # Count significant singular values
    significant_count = np.sum(singular_values > threshold)
    total_count = len(singular_values)
    
    # Calculate spectrum characteristics
    if len(singular_values) > 0:
        max_sv = singular_values[0]
        effective_ratio = significant_count / total_count
        spectrum_decay = singular_values[-1] / max_sv if max_sv > 0 else 0
    else:
        max_sv = 0
        effective_ratio = 0
        spectrum_decay = 0
    
    return {
        'rank': significant_count,
        'total_dimensions': total_count,
        'effective_ratio': effective_ratio,
        'max_singular_value': max_sv,
        'spectrum_decay': spectrum_decay,
        'singular_values': singular_values
    }

def optimize_kr_gr_threshold(states, nreadouts, threshold_range=None):
    """
    Optimize threshold for KR&GR evaluation based on singular value distribution.
    
    Args:
        states (np.array): Reservoir states
        nreadouts (int): Number of readouts
        threshold_range (tuple): (min_threshold, max_threshold) to search
    
    Returns:
        dict: Optimization results with recommended threshold
    """
    if threshold_range is None:
        threshold_range = (1e-6, 1e-1)
    
    # Extract states for analysis
    gr_states = states[:, -1, :]
    kr_states = states[:, -11, :]
    
    # Compute singular values
    _, sGR, _ = np.linalg.svd(gr_states)
    _, sKR, _ = np.linalg.svd(kr_states)
    
    # Test different thresholds
    thresholds = np.logspace(np.log10(threshold_range[0]), 
                            np.log10(threshold_range[1]), 20)
    
    results = []
    for threshold in thresholds:
        kr = np.sum(sKR > threshold)
        gr = np.sum(sGR > threshold)
        cq = kr - gr
        
        results.append({
            'threshold': threshold,
            'KR': kr,
            'GR': gr,
            'CQ': cq
        })
    
    # Find threshold with maximum CQ
    best_idx = np.argmax([r['CQ'] for r in results])
    recommended_threshold = results[best_idx]['threshold']
    
    return {
        'recommended_threshold': recommended_threshold,
        'optimization_results': results,
        'best_performance': results[best_idx]
    }