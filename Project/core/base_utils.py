# -*- coding: utf-8 -*-
"""
Base utility functions for reservoir computing evaluation.

This module contains truly universal utility functions that are shared 
across different evaluation tasks without task-specific modifications.
"""

import numpy as np
import torch

def MSE(pred, desired):
    """
    Calculate Mean Squared Error between prediction and desired output.
    
    Args:
        pred: Predicted values
        desired: Ground truth values
        
    Returns:
        float: Mean squared error
    """
    return np.mean(np.square(np.subtract(pred, desired)))

def NRMSE(pred, y_test, spacer=0.001):
    """
    Calculate Normalized Root Mean Squared Error.
    
    Args:
        pred: Predicted values
        y_test: Test target values  
        spacer: Small value to avoid division by zero
        
    Returns:
        float: Normalized RMSE
    """
    return np.sqrt(MSE(pred, y_test) / np.var(y_test))

def safe_numpy_convert(tensor_or_array):
    """
    Safely convert PyTorch tensors or other formats to numpy arrays.
    
    Args:
        tensor_or_array: Input data (tensor, array, etc.)
        
    Returns:
        np.array: Converted numpy array
    """
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)

def validate_signal_shape(signal, expected_dims=2):
    """
    Validate and reshape signal to expected dimensions.
    
    Args:
        signal: Input signal
        expected_dims: Expected number of dimensions
        
    Returns:
        np.array: Properly shaped signal
    """
    signal = np.asarray(signal)
    if signal.ndim < expected_dims:
        signal = signal.reshape(-1, 1)
    return signal

def safe_divide(numerator, denominator, default_value=0.0):
    """
    Safely divide two numbers, returning default_value if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default_value: Value to return if denominator is zero
        
    Returns:
        float: Division result or default_value
    """
    if np.abs(denominator) < 1e-12:
        return default_value
    return numerator / denominator

def ensure_positive_definite(matrix, regularization=1e-8):
    """
    Ensure a matrix is positive definite by adding regularization.
    
    Args:
        matrix: Input matrix
        regularization: Regularization parameter to add to diagonal
        
    Returns:
        np.array: Regularized matrix
    """
    return matrix + regularization * np.identity(len(matrix))