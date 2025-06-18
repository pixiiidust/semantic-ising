"""
UI export helper functions for Phase 8

Provides temporary file export functionality for Streamlit interface integration.
These functions create temporary files that can be downloaded by users.
"""

import tempfile
import os
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any


def export_csv_results(simulation_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
    """
    Export simulation results as CSV
    
    Args:
        simulation_results: Simulation results dictionary
        analysis_results: Analysis results dictionary
        
    Returns:
        str: Path to temporary CSV file
        
    Raises:
        KeyError: If required data is missing
    """
    # Extract metrics from simulation results (handle both old and new formats)
    if 'metrics' in simulation_results:
        # Old format: metrics nested under 'metrics' key
        metrics = simulation_results['metrics']
    else:
        # New format: metrics directly at top level
        metrics = simulation_results
    
    # Check if we have the required data
    if 'temperatures' not in metrics:
        raise KeyError("simulation_results must contain 'temperatures' key")
    
    # Convert numpy arrays to lists for DataFrame
    csv_data = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            csv_data[key] = value.tolist()
        else:
            csv_data[key] = value
    
    # Add analysis results if available
    if analysis_results and 'anchor_comparison' in analysis_results:
        # Add analysis results as additional columns
        comparison = analysis_results['anchor_comparison']
        for key, value in comparison.items():
            # Repeat the value for each row
            csv_data[f"analysis_{key}"] = [value] * len(next(iter(csv_data.values())))
    
    # Create DataFrame and save to temporary file
    df = pd.DataFrame(csv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


def export_vectors_at_tc(simulation_results: Dict[str, Any]) -> str:
    """
    Export vectors at critical temperature
    
    Args:
        simulation_results: Simulation results dictionary
        
    Returns:
        str: Path to temporary NumPy file
        
    Raises:
        ValueError: If no vector snapshots available
    """
    tc = simulation_results.get('critical_temperature')
    if not tc or 'vector_snapshots' not in simulation_results:
        raise ValueError("No vector snapshots available at Tc")
    
    # Find closest temperature to Tc
    temperatures = list(simulation_results['vector_snapshots'].keys())
    tc_idx = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - tc))
    tc_vectors = simulation_results['vector_snapshots'][temperatures[tc_idx]]
    
    # Save vectors to temporary file
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, tc_vectors)
        return f.name


def export_charts(simulation_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
    """
    Export charts as PNG files
    
    Args:
        simulation_results: Simulation results dictionary
        analysis_results: Analysis results dictionary
        
    Returns:
        str: Path to temporary PNG file (or TXT file if kaleido not available)
    """
    try:
        import plotly.graph_objects as go
        
        # Create a simple chart (placeholder - actual implementation would create specific plots)
        fig = go.Figure()
        
        # Extract metrics (handle both old and new formats)
        if 'metrics' in simulation_results:
            # Old format: metrics nested under 'metrics' key
            metrics = simulation_results['metrics']
        else:
            # New format: metrics directly at top level
            metrics = simulation_results
        
        if 'temperatures' in metrics and 'alignment' in metrics:
            fig.add_trace(go.Scatter(
                x=metrics['temperatures'], 
                y=metrics['alignment'], 
                mode='lines+markers',
                name='Alignment'
            ))
        
        fig.update_layout(title="Alignment vs Temperature")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.write_image(f.name)
            return f.name
            
    except (ImportError, ValueError) as e:
        # If plotly is not available or kaleido is missing, create a simple text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Chart export requires plotly and kaleido packages.\n")
            f.write("Please install them using:\n")
            f.write("pip install plotly kaleido\n")
            f.write(f"Error: {str(e)}\n")
            return f.name


def export_config_file() -> str:
    """
    Export current configuration as YAML
    
    Returns:
        str: Path to temporary YAML file
    """
    config = {
        'temperature_range': [0.1, 3.0],
        'temperature_steps': 50,
        'default_encoder': "sentence-transformers/LaBSE",
        'update_method': "metropolis",
        'simulation_params': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'energy_coupling': 1.0
        },
        'anchor_config': {
            'default_anchor_language': "en",
            'include_anchor_default': False
        },
        'umap_params': {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'random_state': 42
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        return f.name 