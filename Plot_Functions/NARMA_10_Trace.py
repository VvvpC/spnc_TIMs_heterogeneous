#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NARMA-10 Time Series Trace Plotting Script

This script loads NARMA-10 prediction and target data from pickle files
and plots both traces on the same figure for comparison.

Features:
- Flexible filename parameter for easy file switching
- Plots prediction and target traces on the same graph
- High-quality figure output
- Automatic legend and labeling
- Configurable plot styling

Created on 2025-07-25
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import warnings


def load_data_with_compatibility(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load pickle file data with numpy compatibility handling.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the pickle file
        
    Returns:
    --------
    Dict[str, Any]
        Loaded data dictionary
        
    Raises:
    -------
    Exception
        If all loading methods fail
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Try different loading methods for compatibility
    loading_methods = [
        ("Standard pickle.load", lambda f: pickle.load(f)),
        ("Pickle with latin-1 encoding", lambda f: pickle.load(f, encoding='latin-1')),
        ("Pickle with bytes encoding", lambda f: pickle.load(f, encoding='bytes')),
    ]
    
    for method_name, method_func in loading_methods:
        try:
            print(f"Trying: {method_name}")
            with open(filepath, 'rb') as f:
                data = method_func(f)
            print(f"Success with: {method_name}")
            return data
        except Exception as e:
            print(f"Failed with {method_name}: {str(e)[:50]}...")
            continue
    
    raise Exception(f"All loading methods failed for: {filepath}")


def extract_narma_traces(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract prediction and target traces from loaded data.
    
    Parameters:
    -----------
    data : Dict[str, Any]
        Loaded data dictionary
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Prediction and target arrays
        
    Raises:
    -------
    ValueError
        If expected data structure is not found
    """
    # Print debug information
    print(f"Available data keys: {list(data.keys())}")
    
    # Try to find the data in different possible structures
    if 'outputs' in data:
        outputs = data['outputs']
        print(f"Outputs type: {type(outputs)}")
        
        # If outputs is a dict, explore its structure
        if isinstance(outputs, dict):
            print(f"Outputs keys: {list(outputs.keys())}")
            
            # Check for nested outputs structure
            if 'outputs' in outputs:
                inner_outputs = outputs['outputs']
                print(f"Inner outputs type: {type(inner_outputs)}")
                
                if isinstance(inner_outputs, np.ndarray):
                    print(f"Inner outputs shape: {inner_outputs.shape}")
                    # If it's a 2D array, assume rows are prediction and target
                    if inner_outputs.ndim == 2 and inner_outputs.shape[0] == 2:
                        prediction = inner_outputs[0]
                        target = inner_outputs[1]
                        return prediction, target
                    # If it's a 1D array, we might need to split it
                    elif inner_outputs.ndim == 1:
                        # For NARMA-10, typically we have time series data
                        # Try to find if there are separate prediction/target arrays
                        print("Single 1D array found - checking for additional data...")
                        
                elif isinstance(inner_outputs, dict):
                    print(f"Inner outputs dict keys: {list(inner_outputs.keys())}")
                    # Continue searching in the nested dict
                    outputs = inner_outputs
                    
                elif isinstance(inner_outputs, (tuple, list)):
                    print(f"Inner outputs is tuple/list with length: {len(inner_outputs)}")
                    if len(inner_outputs) == 2:
                        # Assume first element is prediction, second is target
                        prediction = np.array(inner_outputs[0])
                        target = np.array(inner_outputs[1])
                        print(f"Extracted prediction shape: {prediction.shape}")
                        print(f"Extracted target shape: {target.shape}")
                        return prediction, target
                    else:
                        # Print info about each element in the tuple
                        for i, element in enumerate(inner_outputs):
                            if isinstance(element, np.ndarray):
                                print(f"  Element {i}: numpy array, shape {element.shape}")
                            else:
                                print(f"  Element {i}: {type(element)}")
                        
                        # If there are numpy arrays, try to use them
                        arrays = [elem for elem in inner_outputs if isinstance(elem, np.ndarray)]
                        if len(arrays) >= 2:
                            prediction = arrays[0]
                            target = arrays[1]
                            print(f"Using first two arrays - prediction shape: {prediction.shape}, target shape: {target.shape}")
                            return prediction, target
            
            # Look for common prediction/target key patterns
            pred_keys = ['prediction', 'pred', 'output', 'y_pred', 'predicted']
            target_keys = ['target', 'actual', 'y_true', 'ground_truth', 'true']
            
            prediction = None
            target = None
            
            for key in pred_keys:
                if key in outputs:
                    prediction = np.array(outputs[key])
                    print(f"Found prediction data with key: {key}, shape: {prediction.shape}")
                    break
                    
            for key in target_keys:
                if key in outputs:
                    target = np.array(outputs[key])
                    print(f"Found target data with key: {key}, shape: {target.shape}")
                    break
                    
            if prediction is not None and target is not None:
                return prediction, target
                
            # If we found outputs as numpy array, try to interpret it
            if isinstance(outputs, np.ndarray):
                print(f"Outputs is numpy array with shape: {outputs.shape}")
                if outputs.ndim == 2:
                    if outputs.shape[0] == 2:
                        # Two rows - assume first is prediction, second is target
                        prediction = outputs[0]
                        target = outputs[1]
                        return prediction, target
                    elif outputs.shape[1] == 2:
                        # Two columns - assume first is prediction, second is target
                        prediction = outputs[:, 0]
                        target = outputs[:, 1]
                        return prediction, target
        
        # If outputs is a numpy array or list with two elements
        elif isinstance(outputs, (np.ndarray, list, tuple)):
            outputs_array = np.array(outputs)
            print(f"Outputs array shape: {outputs_array.shape}")
            if outputs_array.ndim == 2 and outputs_array.shape[0] == 2:
                # Assume first row is prediction, second is target
                prediction = outputs_array[0]
                target = outputs_array[1]
                return prediction, target
            elif outputs_array.ndim == 2 and outputs_array.shape[1] == 2:
                # Assume first column is prediction, second is target
                prediction = outputs_array[:, 0]
                target = outputs_array[:, 1]
                return prediction, target
    
    # Try other common data structures
    if 'prediction' in data and 'target' in data:
        prediction = np.array(data['prediction'])
        target = np.array(data['target'])
        return prediction, target
    
    # If single array, try to split or assume it contains both
    for key in ['data', 'traces', 'time_series']:
        if key in data:
            array_data = np.array(data[key])
            print(f"Found {key} with shape: {array_data.shape}")
            if array_data.ndim == 2 and array_data.shape[0] == 2:
                prediction = array_data[0]
                target = array_data[1]
                return prediction, target
    
    raise ValueError("Could not find prediction and target data in the expected format")


def plot_narma_traces(prediction: np.ndarray, 
                     target: np.ndarray,
                     title: str = "NARMA-10 Prediction vs Target",
                     figsize: Tuple[float, float] = (7, 5),
                     dpi: int = 300,
                     save_path: Optional[Union[str, Path]] = None,
                     show_plot: bool = True) -> plt.Figure:
    """
    Plot NARMA-10 prediction and target traces.
    
    Parameters:
    -----------
    prediction : np.ndarray
        Prediction time series
    target : np.ndarray
        Target time series
    title : str, default="NARMA-10 Prediction vs Target"
        Plot title
    figsize : Tuple[float, float], default=(12, 6)
        Figure size in inches
    dpi : int, default=300
        Figure resolution
    save_path : str or Path, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create time axis
    time_steps = len(prediction)
    time_axis = np.arange(time_steps)
    
    # Plot both traces
    ax.plot(prediction[500:600], label='Target', color='grey', linewidth=1, alpha=0.5)
    ax.plot(target[500:600], label='Prediction', color='purple', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time Steps', fontsize=16)
    ax.set_ylabel('Output Value', fontsize=16)
    
    
    # Add legend
    ax.legend(fontsize=8, loc='upper right')
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Calculate and display performance metrics
    # mse = np.mean((prediction - target) ** 2)
    # mae = np.mean(np.abs(prediction - target))
    
    # # Add text box with metrics
    # metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}'
    # ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
    #         fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_from_file(filename: Union[str, Path],
                  title: Optional[str] = None,
                  save_path: Optional[Union[str, Path]] = None,
                  show_plot: bool = True,
                  **kwargs) -> plt.Figure:
    """
    Load data from file and create NARMA-10 trace plot.
    
    Parameters:
    -----------
    filename : str or Path
        Path to the data file
    title : str, optional
        Custom plot title. If None, will be generated from filename
    save_path : str or Path, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
    **kwargs
        Additional arguments for plot_narma_traces
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    filename = Path(filename)
    
    # Load data
    print(f"Loading data from: {filename}")
    data = load_data_with_compatibility(filename)
    
    # Extract traces
    print("Extracting prediction and target traces...")
    prediction, target = extract_narma_traces(data)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Target shape: {target.shape}")
    
    # Generate title if not provided
    if title is None:
        title = f"NARMA-10 Traces - {filename.stem}"
    
    # Create plot
    fig = plot_narma_traces(
        prediction=prediction,
        target=target,
        title=title,
        save_path=save_path,
        show_plot=show_plot,
        **kwargs
    )
    
    return fig


# Main execution
if __name__ == "__main__":
    # Configuration - easily modifiable filename
    filename = 'results/SingleTests/uniform_bestCQ_NARMA10.pkl'
    
    # Alternative filenames can be easily switched by changing the above line:
    # filename = 'results/SingleTests/uniform_bestMC_NARMA10.pkl'
    # filename = 'results/SingleTests/uniform_bestphase_NARMA10.pkl'
    
    # Output settings
    save_figure = True
    output_filename = f"{Path(filename).stem}_traces.png"
    output_path = Path("Plot_Functions") / output_filename
    
    try:
        # Create the plot
        print("="*60)
        print("NARMA-10 Trace Plotting Script")
        print("="*60)
        
        fig = plot_from_file(
            filename=filename,
            save_path=output_path if save_figure else None,
            show_plot=True,
            figsize=(6, 4),
            dpi=300
        )
        
        print("\nPlot generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()