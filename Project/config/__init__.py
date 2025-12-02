# -*- coding: utf-8 -*-
"""
Configuration module for SPNC evaluation framework.

This module handles environment setup, path configuration, and
global settings for the evaluation framework.
"""

from .paths import setup_environment, get_config

__all__ = ['setup_environment', 'get_config']