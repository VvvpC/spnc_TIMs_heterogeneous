# -*- coding: utf-8 -*-
"""
Framework module for SPNC evaluation orchestration.

This module provides the high-level framework components:
- ReservoirPerformanceEvaluator: Manages parameter scanning and evaluation
- run_evaluation: Unified interface for running evaluations
"""

from .evaluator import ReservoirPerformanceEvaluator
from .runner import run_evaluation

__all__ = ['ReservoirPerformanceEvaluator', 'run_evaluation']