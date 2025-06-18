"""
Export I/O functionality for Phase 8

Provides basic data export functions including JSON, CSV, and embedding export
with comprehensive error handling and validation.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List


def save_json(obj: Any, filepath: str) -> None:
    """
    Store results with proper error handling
    
    Args:
        obj: Object to serialize to JSON
        filepath: Path where to save the JSON file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        raise IOError(f"Failed to save JSON to {filepath}: {e}")


def save_csv(data: Dict[str, List], filepath: str) -> None:
    """
    Store results with proper error handling
    
    Args:
        data: Dictionary with list values to save as CSV
        filepath: Path where to save the CSV file
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise IOError(f"Failed to save CSV to {filepath}: {e}")


def save_embeddings(vectors: np.ndarray, filename: str, filetype: str, output_dir: str) -> str:
    """
    Save embeddings in multiple formats
    
    Args:
        vectors: NumPy array of embeddings to save
        filename: Base filename (without extension)
        filetype: Type identifier for metadata
        output_dir: Directory to save files in
        
    Returns:
        str: Filepath of the saved NumPy file
        
    Raises:
        ValueError: If vectors are invalid
        IOError: If files cannot be written
    """
    # Validate inputs
    if vectors is None:
        raise ValueError("Vectors cannot be None")
    
    if vectors.size == 0:
        raise ValueError("Vectors array cannot be empty")
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise IOError(f"Cannot create output directory {output_dir}: {e}")
    
    filepath_base = os.path.join(output_dir, filename)
    
    # Save as NumPy array
    npy_filepath = f"{filepath_base}.npy"
    try:
        np.save(npy_filepath, vectors)
    except Exception as e:
        raise IOError(f"Failed to save NumPy file {npy_filepath}: {e}")
    
    # Save as parquet for better compression (optional)
    try:
        parquet_filepath = f"{filepath_base}.parquet"
        df = pd.DataFrame(vectors)
        df.to_parquet(parquet_filepath, index=False)
    except ImportError:
        # pandas not available or parquet not supported
        pass
    except Exception as e:
        # Non-critical error, continue without parquet
        pass
    
    # Save metadata
    metadata = {
        'shape': list(vectors.shape),
        'dtype': str(vectors.dtype),
        'filetype': filetype,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_filepath = f"{filepath_base}_metadata.json"
    try:
        save_json(metadata, metadata_filepath)
    except Exception as e:
        raise IOError(f"Failed to save metadata file {metadata_filepath}: {e}")
    
    return npy_filepath 