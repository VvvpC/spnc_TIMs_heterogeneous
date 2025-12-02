#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numpy Pickle File Converter

A utility to convert pickle files between different numpy versions.
Handles compatibility issues when pickle files are created with different numpy versions.

Created on 2025-07-22
@author: Claude Code

Features:
- Convert pickle files from old numpy versions to new versions
- Convert pickle files from new numpy versions to old versions  
- Batch conversion of multiple files
- Safe conversion with backup options
- Detailed error reporting and version information
"""

import pickle
import numpy as np
import sys
import shutil
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import warnings
import traceback
from datetime import datetime
import json


class NumpyPickleConverter:
    """
    Converter for numpy pickle files across different versions.
    
    This class handles the conversion of pickle files that contain numpy arrays
    and may have compatibility issues between different numpy versions.
    """
    
    def __init__(self, create_backup: bool = True, verbose: bool = True):
        """
        Initialize the converter.
        
        Parameters:
        -----------
        create_backup : bool, default=True
            Whether to create backup files before conversion
        verbose : bool, default=True
            Whether to print detailed conversion information
        """
        self.create_backup = create_backup
        self.verbose = verbose
        self.conversion_log = []
        
    def get_numpy_info(self) -> Dict[str, str]:
        """
        Get current numpy version and related information.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary containing numpy version info
        """
        return {
            'numpy_version': np.__version__,
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
        
    def create_backup_file(self, filepath: Union[str, Path]) -> Path:
        """
        Create a backup of the original file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file to backup
            
        Returns:
        --------
        Path
            Path to the backup file
        """
        filepath = Path(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        
        shutil.copy2(filepath, backup_path)
        if self.verbose:
            print(f"Created backup: {backup_path}")
            
        return backup_path
        
    def load_with_compatibility_handling(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load pickle file with various compatibility methods.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the pickle file
            
        Returns:
        --------
        Dict[str, Any]
            Loaded data
            
        Raises:
        -------
        Exception
            If all loading methods fail
        """
        filepath = Path(filepath)
        
        # Try different loading methods
        loading_methods = [
            ("Standard pickle.load", self._load_standard),
            ("Pickle with latin-1 encoding", self._load_latin1),
            ("Pickle with bytes encoding", self._load_bytes),
            ("Pickle with ignore errors", self._load_ignore_errors)
        ]
        
        for method_name, method_func in loading_methods:
            try:
                if self.verbose:
                    print(f"Trying: {method_name}")
                data = method_func(filepath)
                if self.verbose:
                    print(f"✓ Success with: {method_name}")
                return data
            except Exception as e:
                if self.verbose:
                    print(f"✗ Failed with {method_name}: {str(e)[:100]}")
                continue
                
        # If all methods fail, raise the last exception
        raise Exception(f"All loading methods failed for {filepath}")
        
    def _load_standard(self, filepath: Path) -> Dict[str, Any]:
        """Standard pickle loading."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
            
    def _load_latin1(self, filepath: Path) -> Dict[str, Any]:
        """Load with latin-1 encoding (for Python 2 -> 3 compatibility)."""
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin-1')
            
    def _load_bytes(self, filepath: Path) -> Dict[str, Any]:
        """Load with bytes encoding."""
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='bytes')
            
    def _load_ignore_errors(self, filepath: Path) -> Dict[str, Any]:
        """Load with error handling for encoding issues."""
        with open(filepath, 'rb') as f:
            return pickle.load(f, errors='ignore')
            
    def convert_file(self, 
                    input_path: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None,
                    protocol: Optional[int] = None) -> Path:
        """
        Convert a single pickle file to be compatible with current numpy version.
        
        Parameters:
        -----------
        input_path : str or Path
            Input pickle file path
        output_path : str or Path, optional
            Output file path. If None, will overwrite the input file.
        protocol : int, optional
            Pickle protocol version to use. If None, uses HIGHEST_PROTOCOL.
            
        Returns:
        --------
        Path
            Path to the converted file
            
        Raises:
        -------
        FileNotFoundError
            If input file doesn't exist
        Exception
            If conversion fails
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
            
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested and output overwrites input
        backup_path = None
        if self.create_backup and output_path == input_path:
            backup_path = self.create_backup_file(input_path)
            
        conversion_info = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'backup_file': str(backup_path) if backup_path else None,
            'numpy_info': self.get_numpy_info(),
            'success': False,
            'error': None
        }
        
        try:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Converting: {input_path}")
                print(f"Output: {output_path}")
                print(f"Numpy version: {np.__version__}")
                print(f"{'='*60}")
                
            # Load data with compatibility handling
            data = self.load_with_compatibility_handling(input_path)
            
            # Validate and process numpy arrays
            data = self._process_numpy_arrays(data)
            
            # Determine pickle protocol
            if protocol is None:
                protocol = pickle.HIGHEST_PROTOCOL
                
            # Save with current numpy version
            with open(output_path, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
                
            # Verify the conversion worked
            self._verify_conversion(output_path)
            
            conversion_info['success'] = True
            if self.verbose:
                print(f"✓ Conversion successful!")
                print(f"  File size: {input_path.stat().st_size:,} -> {output_path.stat().st_size:,} bytes")
                
        except Exception as e:
            conversion_info['error'] = str(e)
            conversion_info['traceback'] = traceback.format_exc()
            
            if self.verbose:
                print(f"✗ Conversion failed: {e}")
                
            # Clean up output file if conversion failed
            if output_path.exists() and output_path != input_path:
                output_path.unlink()
                
            raise
            
        finally:
            self.conversion_log.append(conversion_info)
            
        return output_path
        
    def _process_numpy_arrays(self, data: Any) -> Any:
        """
        Recursively process numpy arrays in the data structure.
        
        This ensures numpy arrays are compatible with the current version.
        
        Parameters:
        -----------
        data : Any
            Data structure that may contain numpy arrays
            
        Returns:
        --------
        Any
            Processed data structure
        """
        if isinstance(data, np.ndarray):
            # Convert to current numpy version's array format
            return np.array(data, copy=False)
        elif isinstance(data, dict):
            return {key: self._process_numpy_arrays(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            processed = [self._process_numpy_arrays(item) for item in data]
            return type(data)(processed)
        else:
            return data
            
    def _verify_conversion(self, filepath: Path) -> None:
        """
        Verify that the converted file can be loaded successfully.
        
        Parameters:
        -----------
        filepath : Path
            Path to the converted file
            
        Raises:
        -------
        Exception
            If verification fails
        """
        try:
            with open(filepath, 'rb') as f:
                pickle.load(f)
        except Exception as e:
            raise Exception(f"Conversion verification failed: {e}")
            
    def convert_batch(self, 
                     input_directory: Union[str, Path],
                     output_directory: Optional[Union[str, Path]] = None,
                     pattern: str = "*.pkl",
                     protocol: Optional[int] = None) -> List[Path]:
        """
        Convert multiple pickle files in a directory.
        
        Parameters:
        -----------
        input_directory : str or Path
            Directory containing input files
        output_directory : str or Path, optional
            Directory for output files. If None, files are converted in place.
        pattern : str, default="*.pkl"
            File pattern to match
        protocol : int, optional
            Pickle protocol to use
            
        Returns:
        --------
        List[Path]
            List of successfully converted file paths
        """
        input_dir = Path(input_directory)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        if output_directory is not None:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None
            
        # Find matching files
        input_files = list(input_dir.glob(pattern))
        if not input_files:
            if self.verbose:
                print(f"No files matching pattern '{pattern}' found in {input_dir}")
            return []
            
        if self.verbose:
            print(f"Found {len(input_files)} files to convert")
            
        converted_files = []
        failed_files = []
        
        for input_file in input_files:
            try:
                if output_dir is not None:
                    output_file = output_dir / input_file.name
                else:
                    output_file = None
                    
                converted_path = self.convert_file(input_file, output_file, protocol)
                converted_files.append(converted_path)
                
            except Exception as e:
                failed_files.append((input_file, str(e)))
                if self.verbose:
                    print(f"Failed to convert {input_file}: {e}")
                    
        if self.verbose:
            print(f"\nBatch conversion complete:")
            print(f"  Successful: {len(converted_files)}")
            print(f"  Failed: {len(failed_files)}")
            
        return converted_files
        
    def save_conversion_log(self, log_path: Union[str, Path]) -> None:
        """
        Save the conversion log to a JSON file.
        
        Parameters:
        -----------
        log_path : str or Path
            Path to save the log file
        """
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversion_log, f, indent=2, default=str)
            
        if self.verbose:
            print(f"Conversion log saved to: {log_path}")


# Convenience functions
def convert_numpy_pickle(input_file: Union[str, Path], 
                        output_file: Optional[Union[str, Path]] = None,
                        create_backup: bool = True) -> Path:
    """
    Quick function to convert a single numpy pickle file.
    
    Parameters:
    -----------
    input_file : str or Path
        Input pickle file
    output_file : str or Path, optional
        Output file path
    create_backup : bool, default=True
        Whether to create a backup
        
    Returns:
    --------
    Path
        Path to the converted file
    """
    converter = NumpyPickleConverter(create_backup=create_backup)
    return converter.convert_file(input_file, output_file)


def batch_convert_numpy_pickles(directory: Union[str, Path],
                               pattern: str = "*.pkl",
                               create_backup: bool = True) -> List[Path]:
    """
    Quick function to convert multiple numpy pickle files.
    
    Parameters:
    -----------
    directory : str or Path
        Directory containing files to convert
    pattern : str, default="*.pkl"
        File pattern to match
    create_backup : bool, default=True
        Whether to create backups
        
    Returns:
    --------
    List[Path]
        List of converted file paths
    """
    converter = NumpyPickleConverter(create_backup=create_backup)
    return converter.convert_batch(directory, pattern=pattern)


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert numpy pickle files for version compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python numpy_pickle_converter.py data.pkl
  
  # Convert to different output file
  python numpy_pickle_converter.py data.pkl -o data_converted.pkl
  
  # Batch convert all pkl files in directory
  python numpy_pickle_converter.py --batch /path/to/data/
  
  # Convert without backup
  python numpy_pickle_converter.py data.pkl --no-backup
        """
    )
    
    parser.add_argument("input", help="Input file or directory path")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("--batch", action="store_true", help="Batch convert files in directory")
    parser.add_argument("--pattern", default="*.pkl", help="File pattern for batch mode (default: *.pkl)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    parser.add_argument("--protocol", type=int, help="Pickle protocol version to use")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--log", help="Save conversion log to file")
    
    args = parser.parse_args()
    
    # Create converter
    converter = NumpyPickleConverter(
        create_backup=not args.no_backup,
        verbose=not args.quiet
    )
    
    try:
        if args.batch:
            # Batch conversion
            converted_files = converter.convert_batch(
                input_directory=args.input,
                output_directory=args.output,
                pattern=args.pattern,
                protocol=args.protocol
            )
            print(f"\nConverted {len(converted_files)} files successfully")
        else:
            # Single file conversion
            converted_path = converter.convert_file(
                input_path=args.input,
                output_path=args.output,
                protocol=args.protocol
            )
            print(f"\nFile converted successfully: {converted_path}")
            
        # Save log if requested
        if args.log:
            converter.save_conversion_log(args.log)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)