# -*- coding: utf-8 -*-
"""
TI46 speech recognition task module.

This module provides evaluation of speech recognition capability
using the TI46 digit recognition benchmark.
"""

from .evaluator import evaluate_Ti46

__all__ = ['evaluate_Ti46']