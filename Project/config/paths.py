# -*- coding: utf-8 -*-
"""
Path configuration and environment setup for SPNC evaluation framework.

This module handles the complex repository path discovery and environment
setup that was previously embedded in the main evaluation script.
"""

import os
from pathlib import Path

class Config:
    """
    Configuration management for SPNC evaluation framework.
    
    Handles path discovery, repository setup, and environment configuration
    that enables the framework to find required dependencies.
    """
    
    def __init__(self):
        # Define candidate paths for repository discovery
        self.CANDIDATES = [
            Path(r"C:\Users\tom\Desktop\Repository"),
            Path(r"C:\Users\Chen\Desktop\Repository"),
            Path(r"/Users/vvvp./Desktop"),
        ]
        
        # Repositories to search for
        self.repos = ('machine_learning_library',)
        
        # Find existing search paths
        self.searchpaths = [p for p in self.CANDIDATES if p.exists()]
        
        # Environment setup status
        self._environment_setup = False
    
    def setup_environment(self):
        """
        Set up the Python environment for SPNC evaluation.
        
        This method configures the Python path to enable imports of
        required machine learning and reservoir computing libraries.
        """
        if self._environment_setup:
            return  # Already configured
        
        try:
            # Import repository tools for path setup
            import repo_tools
            repo_tools.repos_path_finder(self.searchpaths, self.repos)
            
            print(f"Environment configured with search paths: {self.searchpaths}")
            self._environment_setup = True
            
        except ImportError as e:
            print(f"Warning: Could not import repo_tools. Some functionality may be limited: {e}")
        except Exception as e:
            print(f"Warning: Environment setup encountered issues: {e}")
    
    def get_search_paths(self):
        """Get the list of configured search paths."""
        return self.searchpaths
    
    def add_search_path(self, path):
        """
        Add an additional search path.
        
        Args:
            path (str or Path): Path to add to search paths
        """
        path = Path(path)
        if path.exists() and path not in self.searchpaths:
            self.searchpaths.append(path)
            # Re-setup environment if it was already configured
            if self._environment_setup:
                self._environment_setup = False
                self.setup_environment()
    
    def validate_environment(self):
        """
        Validate that required dependencies are available.
        
        Returns:
            dict: Validation results with status and missing dependencies
        """
        required_modules = [
            'spnc',
            'single_node_res', 
            'deterministic_mask',
            'spnc_ml',
            'ridge_regression',
            'linear_layer',
            'mask',
            'utility',
            'NARMA10'
        ]
        
        available = []
        missing = []
        
        for module in required_modules:
            try:
                __import__(module)
                available.append(module)
            except ImportError:
                missing.append(module)
        
        return {
            'status': 'complete' if not missing else 'partial' if available else 'failed',
            'available': available,
            'missing': missing,
            'total_required': len(required_modules)
        }

# Global configuration instance
_config = Config()

def setup_environment():
    """
    Set up the environment for SPNC evaluation framework.
    
    This is the main function to call before using the framework.
    It configures paths and validates the environment.
    """
    _config.setup_environment()
    
    # Validate setup
    validation = _config.validate_environment()
    if validation['status'] != 'complete':
        print(f"Environment validation: {validation['status']}")
        if validation['missing']:
            print(f"Missing modules: {validation['missing']}")
        print("Some functionality may be limited.")
    else:
        print("Environment setup completed successfully.")

def get_config():
    """
    Get the global configuration instance.
    
    Returns:
        Config: Global configuration object
    """
    return _config

def get_search_paths():
    """Get configured search paths."""
    return _config.get_search_paths()

def add_search_path(path):
    """Add a search path to the configuration."""
    _config.add_search_path(path)