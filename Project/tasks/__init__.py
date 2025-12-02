# -*- coding: utf-8 -*-
"""
Task evaluation modules for SPNC performance assessment.

Each task module is self-contained and includes:
- Signal generation specific to the task
- Processing functions optimized for the task
- Main evaluation function

Available tasks:
- memory_capacity: Linear memory capacity evaluation
- kr_gr: Kernel Rank and Generalization Rank evaluation  
- narma10: NARMA10 nonlinear system identification
- ti46: TI46 speech recognition task
"""

from .memory_capacity import evaluate_MC
from .kr_gr import evaluate_KRandGR  
from .narma10 import evaluate_NARMA10
from .ti46 import evaluate_Ti46

__all__ = [
    'evaluate_MC',
    'evaluate_KRandGR', 
    'evaluate_NARMA10',
    'evaluate_Ti46'
]