"""
Pareto Front Plotting Module - Optimized Version

This module provides functionality to visualize Pareto fronts and related data points
from optimization trials. It supports flexible file loading by filename and multi-file
compatibility for future extensions.

Author: Generated for SPNC Optuna Heterogeneous project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Union, Dict
import warnings
warnings.filterwarnings('ignore')

class ParetoFrontPlotter:
    """
    A class for plotting Pareto fronts and analyzing dominated solutions.
    Supports flexible file loading and multi-file compatibility.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the ParetoFrontPlotter.
        
        Args:
            data_dir (str): Directory containing the CSV data files. If None, uses current directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.pareto_front_df = None
        self.all_trials_df = None
        self.loaded_files = {}  # Track loaded files for multi-file support
    
    def list_available_files(self, pattern: str = "*.csv") -> List[Path]:
        """
        List all available CSV files in the data directory.
        
        Args:
            pattern (str): File pattern to search for (default: "*.csv")
            
        Returns:
            List[Path]: List of available files
        """
        files = list(self.data_dir.glob(pattern))
        if files:
            print(f"Available files in {self.data_dir}:")
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file.name}")
        else:
            print(f"No files matching '{pattern}' found in {self.data_dir}")
        return files
    
    def load_data_by_filename(self, 
                             pareto_filename: str,
                             all_trials_filename: Optional[str] = None) -> None:
        """
        Load data from specific filenames.
        
        Args:
            pareto_filename (str): Name of the Pareto front data file
            all_trials_filename (str, optional): Name of all trials data file.
                                                If None, uses pareto_filename for both.
        """
        try:
            # Load Pareto front data
            pareto_path = self.data_dir / pareto_filename
            if not pareto_path.exists():
                raise FileNotFoundError(f"Pareto file not found: {pareto_path}")
            
            print(f"Loading Pareto data from: {pareto_filename}")
            self.pareto_front_df = self._load_and_process_file(pareto_path, "pareto")
            self.loaded_files['pareto'] = pareto_filename
            
            # Load all trials data
            if all_trials_filename:
                all_trials_path = self.data_dir / all_trials_filename
                if not all_trials_path.exists():
                    raise FileNotFoundError(f"All trials file not found: {all_trials_path}")
                
                print(f"Loading all trials data from: {all_trials_filename}")
                self.all_trials_df = self._load_and_process_file(all_trials_path, "all_trials")
                self.loaded_files['all_trials'] = all_trials_filename
            else:
                # Use Pareto data for all trials if not specified
                print("Using Pareto data for all trials (no separate all_trials file specified)")
                self.all_trials_df = self.pareto_front_df.copy()
                self.loaded_files['all_trials'] = pareto_filename
            
            print(f"Successfully loaded {len(self.pareto_front_df)} Pareto front points")
            print(f"Successfully loaded {len(self.all_trials_df)} total trial points")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def load_multiple_files(self, file_configs: List[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple files for comparison or combined analysis.
        
        Args:
            file_configs (List[Dict]): List of file configurations, each containing:
                - 'name': identifier for the dataset
                - 'pareto_file': filename for Pareto data
                - 'all_trials_file': (optional) filename for all trials data
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded datasets
        
        Example:
            configs = [
                {'name': 'study1', 'pareto_file': 'study1_pareto.csv'},
                {'name': 'study2', 'pareto_file': 'study2_pareto.csv', 'all_trials_file': 'study2_all.csv'}
            ]
        """
        datasets = {}
        
        for config in file_configs:
            name = config['name']
            pareto_file = config['pareto_file']
            all_trials_file = config.get('all_trials_file')
            
            try:
                print(f"\nLoading dataset '{name}'...")
                
                # Load Pareto data
                pareto_path = self.data_dir / pareto_file
                if not pareto_path.exists():
                    print(f"Warning: Pareto file not found for {name}: {pareto_path}")
                    continue
                
                pareto_df = self._load_and_process_file(pareto_path, "pareto")
                
                # Load all trials data
                if all_trials_file:
                    all_trials_path = self.data_dir / all_trials_file
                    if all_trials_path.exists():
                        all_trials_df = self._load_and_process_file(all_trials_path, "all_trials")
                    else:
                        print(f"Warning: All trials file not found for {name}, using Pareto data")
                        all_trials_df = pareto_df.copy()
                else:
                    all_trials_df = pareto_df.copy()
                
                datasets[name] = {
                    'pareto_front': pareto_df,
                    'all_trials': all_trials_df,
                    'files': {
                        'pareto': pareto_file,
                        'all_trials': all_trials_file or pareto_file
                    }
                }
                
                print(f"  Loaded {len(pareto_df)} Pareto points, {len(all_trials_df)} total points")
                
            except Exception as e:
                print(f"Error loading dataset '{name}': {e}")
                continue
        
        self.loaded_files['datasets'] = datasets
        return datasets
    
    def _load_and_process_file(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """
        Load and process a single CSV file, handling different formats.
        
        Args:
            file_path (Path): Path to the CSV file
            file_type (str): Type of file ('pareto' or 'all_trials')
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        print(f"Loading file: {file_path}")
        raw_df = pd.read_csv(file_path)
        
        print(f"File shape: {raw_df.shape}")
        print(f"Columns: {list(raw_df.columns)}")
        
        # Check if this is the new format (has 'values' column)
        if 'values' in raw_df.columns:
            print("Detected new format (with 'values' column)")
            # Show sample of values column for debugging
            print("Sample values:")
            for i, val in enumerate(raw_df['values'].head(3)):
                print(f"  Row {i}: {val} (type: {type(val)})")
            return self._process_new_format(raw_df)
        else:
            print("Detected legacy format")
            return self._process_legacy_format(raw_df)
    
    def _process_new_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process new format CSV with 'values' column."""
        import ast
        
        df = raw_df.copy()
        
        # Parse the values column to extract CQ and MC with error handling
        def parse_values(x):
            """Safely parse values column."""
            if pd.isna(x):
                return None
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse values: {x}")
                    return None
            elif isinstance(x, (list, tuple)):
                return x
            else:
                # Handle single numeric values or other types
                print(f"Warning: Unexpected value type in 'values' column: {type(x)} - {x}")
                return None
        
        values_parsed = raw_df['values'].apply(parse_values)
        
        # Extract CQ and MC with safe indexing
        def safe_extract(x, index, default=None):
            """Safely extract value from list/tuple."""
            if x is None:
                return default
            try:
                if isinstance(x, (list, tuple)) and len(x) > index:
                    return x[index]
                else:
                    return default
            except (IndexError, TypeError):
                return default
        
        df['CQ'] = values_parsed.apply(lambda x: safe_extract(x, 0))
        df['MC'] = values_parsed.apply(lambda x: safe_extract(x, 1))
        
        # Check for any missing CQ or MC values
        missing_cq = df['CQ'].isna().sum()
        missing_mc = df['MC'].isna().sum()
        if missing_cq > 0 or missing_mc > 0:
            print(f"Warning: Found {missing_cq} missing CQ values and {missing_mc} missing MC values")
            print("Rows with missing values will be dropped")
            df = df.dropna(subset=['CQ', 'MC'])
        
        # Extract parameter columns (remove param_ prefix)
        param_cols = [col for col in raw_df.columns if col.startswith('param_')]
        for col in param_cols:
            new_col_name = col.replace('param_', '')
            df[new_col_name] = raw_df[col]
        
        return df
    
    def _process_legacy_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process legacy format CSV."""
        # Assume legacy format already has CQ and MC columns
        if 'CQ' not in raw_df.columns or 'MC' not in raw_df.columns:
            print("Warning: Legacy format file missing CQ or MC columns")
        return raw_df
    
    def load_data(self) -> None:
        """
        Load data using automatic detection (legacy method for backward compatibility).
        This method is deprecated in favor of load_data_by_filename().
        """
        print("Warning: load_data() is deprecated. Consider using load_data_by_filename() for better control.")
        
        # Try to find files automatically
        pareto_files = list(self.data_dir.glob("*pareto.csv"))
        
        if pareto_files:
            # Use the first found pareto file
            pareto_filename = pareto_files[0].name
            print(f"Auto-detected Pareto file: {pareto_filename}")
            
            # Look for corresponding all_trials file
            base_name = pareto_filename.replace('_pareto.csv', '').replace('pareto.csv', '')
            possible_all_trials = [
                f"{base_name}_all_trials.csv",
                f"{base_name}_all.csv",
                "all_trials.csv"
            ]
            
            all_trials_filename = None
            for filename in possible_all_trials:
                if (self.data_dir / filename).exists():
                    all_trials_filename = filename
                    break
            
            self.load_data_by_filename(pareto_filename, all_trials_filename)
        else:
            # Try legacy format
            if (self.data_dir / "pareto_front.csv").exists():
                self.load_data_by_filename("pareto_front.csv", "all_trials.csv")
            else:
                # List available files to help user
                print("No Pareto files found. Available files:")
                self.list_available_files()
                raise FileNotFoundError("No suitable Pareto files found. Please use load_data_by_filename() to specify files explicitly.")
    
    def find_near_pareto_points(self, distance_threshold: float = 0.5, 
                               max_points: int = 50) -> pd.DataFrame:
        """
        Find points from all trials that are close to the Pareto front.
        
        Args:
            distance_threshold (float): Maximum distance to Pareto front
            max_points (int): Maximum number of near-Pareto points to return
            
        Returns:
            pd.DataFrame: Points near the Pareto front
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data_by_filename() first.")
        
        # Extract CQ and MC values for distance calculation
        pareto_points = self.pareto_front_df[['CQ', 'MC']].values
        all_points = self.all_trials_df[['CQ', 'MC']].values
        
        # Normalize the data for distance calculation
        cq_range = all_points[:, 0].max() - all_points[:, 0].min()
        mc_range = all_points[:, 1].max() - all_points[:, 1].min()
        
        pareto_normalized = pareto_points.copy()
        pareto_normalized[:, 0] = (pareto_points[:, 0] - all_points[:, 0].min()) / cq_range
        pareto_normalized[:, 1] = (pareto_points[:, 1] - all_points[:, 1].min()) / mc_range
        
        all_normalized = all_points.copy()
        all_normalized[:, 0] = (all_points[:, 0] - all_points[:, 0].min()) / cq_range
        all_normalized[:, 1] = (all_points[:, 1] - all_points[:, 1].min()) / mc_range
        
        # Calculate minimum distance from each point to the Pareto front
        distances = cdist(all_normalized, pareto_normalized, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Find points within distance threshold, excluding Pareto front points
        pareto_numbers = set(self.pareto_front_df['number'].values)
        near_indices = []
        
        for i, (distance, trial_number) in enumerate(zip(min_distances, self.all_trials_df['number'])):
            if distance <= distance_threshold and trial_number not in pareto_numbers:
                near_indices.append(i)
        
        # Sort by distance and limit to max_points
        near_indices = sorted(near_indices, key=lambda i: min_distances[i])[:max_points]
        
        return self.all_trials_df.iloc[near_indices].copy()
    
    def plot_pareto_front_2d(self, figsize: Tuple[int, int] = (8, 5),
                            distance_threshold: float = 0.5,
                            max_near_points: int = 50,
                            save_path: Optional[str] = None,
                            title_suffix: str = "") -> plt.Figure:
        """
        Create a 2D plot of the Pareto front with CQ vs MC.
        
        Args:
            figsize (tuple): Figure size (width, height)
            distance_threshold (float): Distance threshold for near-Pareto points
            max_near_points (int): Maximum number of near-Pareto points to show
            save_path (str, optional): Path to save the figure
            title_suffix (str): Additional text to add to the title
            
        Returns:
            plt.Figure: The created figure
        """
        if self.pareto_front_df is None or self.all_trials_df is None:
            raise ValueError("Data not loaded. Call load_data_by_filename() first.")
        
        # Find near-Pareto points
        near_pareto_df = self.find_near_pareto_points(distance_threshold, max_near_points)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all trials as background points
        # ax.scatter(self.all_trials_df['CQ'], self.all_trials_df['MC'], 
        #           alpha=0.3, s=20, c='lightgray', label=f'All Trials (n={len(self.all_trials_df)})')
        
        # Plot near-Pareto points
        if len(near_pareto_df) > 0:
            ax.scatter(near_pareto_df['CQ'], near_pareto_df['MC'], 
                      alpha=0.5, s=40, c='blue', label=f'Dominated Points')
        
        # Plot Pareto front
        pareto_sorted = self.pareto_front_df.sort_values('CQ')
        # ax.plot(pareto_sorted['CQ'], pareto_sorted['MC'], 
        #        'r-', linewidth=2, alpha=0.7, label='Pareto Front Connection')
        ax.scatter(self.pareto_front_df['CQ'], self.pareto_front_df['MC'], 
                  s=400, c='green', linewidth=1, 
                  marker='*', label=f'Pareto Front', zorder=5)
        
        # Annotate Pareto front points with their trial numbers
        # for _, row in self.pareto_front_df.iterrows():
        #     # 为了尽量避免重叠，采用交错的xytext偏移和对齐方式
        #     idx = list(self.pareto_front_df.index).index(row.name)
        #     # 交错偏移和对齐
        #     offset_options = [
        #         ((8, 8), 'left', 'bottom'),
        #         ((-8, 8), 'right', 'bottom'),
        #         ((8, -8), 'left', 'top'),
        #         ((-8, -8), 'right', 'top'),
        #         ((0, 15), 'center', 'bottom'),
        #         ((0, -15), 'center', 'top'),
        #     ]
        #     offset, ha, va = offset_options[idx % len(offset_options)]
        #     ax.annotate(
        #         f'{row["CQ"]:.0f}, {row["MC"]:.2f}',
        #         (row['CQ'], row['MC']),
        #         xytext=offset, textcoords='offset points',
        #         fontsize=10, alpha=0.95, fontweight='bold', color='darkgreen',
        #         ha=ha, va=va,
        #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", lw=0.8, alpha=0.7)
        #     )
        
        # Formatting
        ax.set_xlabel('Minimum NRMSE', fontsize=16)
        ax.set_ylabel('Average NRMSE', fontsize=16)
        # xiufu
        # The original line is incorrect usage of set_ticklabels and 'xlabel' is undefined.
        # If the intent is to set tick label font size, use tick_params:
        ax.tick_params(axis='both', labelsize=14)

        
        
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add file info and distance threshold to the plot
        
        plt.tight_layout()
        
        # Auto-generate save_path based on loaded pareto filename if not provided
        if save_path is None and 'pareto' in self.loaded_files:
            pareto_filename = self.loaded_files['pareto']
            save_path = pareto_filename.replace('.csv', '.png')
            print(f"Auto-generated save path: {save_path}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_multiple_studies_comparison(self, datasets: Dict[str, Dict], 
                                       figsize: Tuple[int, int] = (8, 5),
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple studies for comparison.
        
        Args:
            datasets (Dict): Dictionary of datasets from load_multiple_files()
            figsize (tuple): Figure size (width, height)
            save_path (str, optional): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        # 从study_name中提取label，'name': 'Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049'，label为在Pareto_后面的字符串
        labels = [name.split('_')[5] for name in datasets.keys()]
   

        if not datasets:
            raise ValueError("No datasets provided for comparison")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette for different studies
        colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
        
        for i, (name, data) in enumerate(datasets.items()):
            pareto_df = data['pareto_front']
            color = colors[i]
            
            # Plot Pareto front
            pareto_sorted = pareto_df.sort_values('CQ')

            ax.scatter(pareto_df['CQ'], pareto_df['MC'], label=labels[i], 
                      s=80, c=color, linewidth=0.5, 
                      alpha=0.8, zorder=5)
        
        # Formatting
        ax.set_xlabel('Computational Quality (CQ)', fontsize=16)
        ax.set_ylabel('Memory Capacity (MC)', fontsize=16)
        # ax.set_title('Multi-Study Pareto Front Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        return fig


def main():
    """Main function demonstrating the usage of the optimized ParetoFrontPlotter."""
    
    # Example 1: Load specific files by name
    # print("=== Example 1: Loading specific files ===")
    plotter = ParetoFrontPlotter(data_dir="saved_studies")  # or your data directory
    
    # List available files first
    plotter.list_available_files()
    
    # Load specific files (modify these names according to your files)
    try:
        plotter.load_data_by_filename(
            pareto_filename="TIMs_Hetero_tempsweep_Task_Pareto_20251208_105813_pareto.csv",  # Replace with your file
            all_trials_filename="TIMs_Hetero_tempsweep_Task_Pareto_20251208_105813_trials.csv"  # Optional
        )
        
        # Create 2D plot (save_path will be auto-generated from pareto filename)
        fig1 = plotter.plot_pareto_front_2d(
            distance_threshold=0.3,
            max_near_points=30,
            # save_path will be auto-generated as: CQ_MC_Pareto_SoftGate_th01_beta50_20250905_123602_pareto.png
            title_suffix="Specific File Load"
        )
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Specific files not found: {e}")
        print("Falling back to automatic detection...")
        
        # Fallback to automatic detection
        plotter.load_data()
        fig1 = plotter.plot_pareto_front_2d(
            distance_threshold=0.3,
            max_near_points=30
            # save_path will be auto-generated from detected pareto filename
        )
    
    # # Example 2: Load multiple files for comparison
    # print("\n=== Example 2: Multi-file comparison ===")
    # file_configs = [
    #     {
    #         'name': 'Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049', 
    #         'pareto_file': 'Reservoir_Morphology_CQ_MC_Pareto_uniform_2_20250725_104049_pareto.csv',
    #     },
    #     {
    #         'name': 'Reservoir_Morphology_CQ_MC_Pareto_normaldistribution_20250725_104555', 
    #         'pareto_file': 'Reservoir_Morphology_CQ_MC_Pareto_normaldistribution_20250725_104555_pareto.csv',
    #     },
    #     {
    #         'name': 'Reservoir_Morphology_CQ_MC_Pareto_random_20250725_105052', 
    #         'pareto_file': 'Reservoir_Morphology_CQ_MC_Pareto_random_20250725_105052_pareto.csv',

    #     }
    #     # Add more studies as needed
    # ]
    
    # try:
    #     datasets = plotter.load_multiple_files(file_configs)
    #     if datasets:
    #         fig2 = plotter.plot_multiple_studies_comparison(
    #             datasets=datasets,
    #             save_path="multi_study_comparison.png"
    #         )
    # except Exception as e:
    #     print(f"Multi-file loading failed: {e}")
    
    # # Show plots
    # plt.show()
    
    print("\nPlotting complete! Check the generated PNG files.")


if __name__ == "__main__":
    main()