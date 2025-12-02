# -*- coding: utf-8 -*-
"""
Core module for SPNC task-independent metrics evaluation framework.

This module provides the fundamental components:
- ReservoirParams: Parameter management for reservoir computing
- RunSpnc: Core reservoir computing execution
- Base utilities: Common mathematical and utility functions
"""

from .reservoir import ReservoirParams, RunSpnc
from .base_utils import MSE, NRMSE, safe_numpy_convert, validate_signal_shape

__all__ = [
    'ReservoirParams', 
    'RunSpnc',
    'MSE', 
    'NRMSE', 
    'safe_numpy_convert', 
    'validate_signal_shape'
]