# -*- coding: utf-8 -*-
"""
Core reservoir computing components for SPNC evaluation framework.

This module provides:
- ReservoirParams: Comprehensive parameter management for SPNC systems
- RunSpnc: Unified interface for reservoir computing execution
"""

import numpy as np
from spnc import spnc_anisotropy
from single_node_res import single_node_reservoir
from deterministic_mask import fixed_seed_mask, max_sequences_mask

class ReservoirParams:
    """
    Comprehensive parameter management for superparamagnetic nanodot 
    reservoir computing systems.
    
    This class encapsulates both physical parameters of the SPNC system
    and computational parameters for the reservoir network.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize reservoir parameters with default values.
        
        Args:
            **kwargs: Parameter overrides
                Physical parameters: h, theta_H, k_s_0, phi, beta_prime
                Network parameters: Nvirt, m0, bias, Nwarmup
                Computational parameters: theta, gamma, etc.
        """
        # Physical parameters of the superparamagnetic nanodot
        self.h = kwargs.get('h', 0.4)
        self.theta_H = kwargs.get('theta_H', 90)
        self.k_s_0 = kwargs.get('k_s_0', 0)
        self.phi = kwargs.get('phi', 45)
        self.beta_prime = kwargs.get('beta_prime', 20)
        
        # Network topology parameters
        self.Nvirt = kwargs.get('Nvirt', 30)
        self.m0 = kwargs.get('m0', 0.007586422893538462)
        self.bias = kwargs.get('bias', True)
        self.Nwarmup = kwargs.get('Nwarmup', 0)
        self.verbose_repr = kwargs.get('verbose_repr', False)
        
        # Computational parameters dictionary
        self.params = {
            'theta': kwargs.get('theta', 0.5540233436467944),
            'gamma': kwargs.get('gamma', 0.13738441393289658),
            'delay_feedback': kwargs.get('delay_feedback', 0),
            'Nvirt': self.Nvirt,
            'length_warmup': self.Nwarmup,
            'warmup_sample': self.Nwarmup * self.Nvirt,
            'voltage_noise': kwargs.get('voltage_noise', False),
            'seed_voltage_noise': kwargs.get('seed_voltage_noise', 1234),
            'delta_V': kwargs.get('delta_V', 0.1),
            'johnson_noise': kwargs.get('johnson_noise', False),
            'seed_johnson_noise': kwargs.get('seed_johnson_noise', 1234),
            'mean_johnson_noise': kwargs.get('mean_johnson_noise', 0.0000),
            'std_johnson_noise': kwargs.get('std_johnson_noise', 0.00001),
            'thermal_noise': kwargs.get('thermal_noise', False),
            'seed_thermal_noise': kwargs.get('seed_thermal_noise', 1234),
            'lambda_ou': kwargs.get('lambda_ou', 1.0),
            'sigma_ou': kwargs.get('sigma_ou', 0.1)
        }
        
        # Apply custom parameter updates
        for key in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 
                    'Nvirt', 'm0', 'bias', 'Nwarmup']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        if 'params' in kwargs and isinstance(kwargs['params'], dict):
            self.params.update(kwargs['params'])
    
    def update_params(self, **kwargs):
        """
        Safely update reservoir parameters.
        
        Args:
            **kwargs: Parameters to update
                - Direct attributes: h, theta_H, k_s_0, phi, beta_prime, etc.
                - Params dict entries: theta, gamma, etc.
        
        Raises:
            AttributeError: If attempting to set unknown parameter
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Also update in params dict if it exists there
                if key in self.params:
                    self.params[key] = value
            elif key in self.params:
                self.params[key] = value
            else:
                # For extensibility, we warn instead of raising error
                print(f"Warning: Unknown parameter '{key}' ignored")
    
    def print_params(self, verbose=False):
        """
        Print current parameter configuration.
        
        Args:
            verbose (bool): If True, print detailed parameter information
        """
        if not verbose:
            print(f"ReservoirParams(h={self.h}, beta_prime={self.beta_prime}, "
                  f"Nvirt={self.Nvirt}, m0={self.m0:.6f})")
        else:
            print(f"ReservoirParams detailed configuration:")
            print(f"  Physical parameters:")
            print(f"    h = {self.h}")
            print(f"    theta_H = {self.theta_H}")
            print(f"    k_s_0 = {self.k_s_0}")
            print(f"    phi = {self.phi}")
            print(f"    beta_prime = {self.beta_prime}")
            print(f"  Network parameters:")
            print(f"    Nvirt = {self.Nvirt}")
            print(f"    m0 = {self.m0}")
            print(f"    bias = {self.bias}")
            print(f"  Computational parameters:")
            for k, v in self.params.items():
                print(f"    {k}: {v}")
    
    def copy(self):
        """
        Create a deep copy of the reservoir parameters.
        
        Returns:
            ReservoirParams: Deep copy of current parameters
        """
        new_params = ReservoirParams()
        
        # Copy direct attributes
        for attr in ['h', 'theta_H', 'k_s_0', 'phi', 'beta_prime', 
                     'Nvirt', 'm0', 'bias', 'Nwarmup', 'verbose_repr']:
            setattr(new_params, attr, getattr(self, attr))
        
        # Copy params dictionary
        new_params.params = self.params.copy()
        
        return new_params

def RunSpnc(signal, Nin, Nout, Nvirt, m0, transform, params, **kwargs):
    """
    Run reservoir computing with the superparamagnetic nanodot system.
    
    This is the unified interface for executing reservoir computing across
    all evaluation tasks. It handles mask generation and signal transformation.
    
    Args:
        signal: Input signal sequence
        Nin: Number of input nodes
        Nout: Number of output nodes  
        Nvirt: Number of virtual nodes
        m0: Mask scaling parameter
        transform: SPNC transformation function
        params: Computational parameters dictionary
        **kwargs: Additional options
            fixed_mask (bool): Use deterministic mask
            seed_mask (int): Seed for mask generation (>=0) or -1 for max_sequences
    
    Returns:
        np.array: Reservoir state outputs
    """
    # Create single node reservoir
    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res=transform)
    
    # Handle mask configuration
    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask:
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask >= 0:
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            # Use max_sequences_mask when seed_mask < 0
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    
    # Execute reservoir transformation
    S, _ = snr.transform(signal, params)
    
    return S