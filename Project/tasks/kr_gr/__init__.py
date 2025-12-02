# -*- coding: utf-8 -*-
"""
Kernel Rank and Generalization Rank evaluation task module.

This module provides evaluation of computational quality through
KR (Kernel Rank) and GR (Generalization Rank) metrics.
"""

from .evaluator import evaluate_KRandGR

__all__ = ['evaluate_KRandGR']