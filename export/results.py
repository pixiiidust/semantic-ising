"""
Comprehensive results export functionality for Phase 8

Provides complete export functionality for simulation results including
anchor comparison metrics and experiment configuration.
"""

import os
from typing import Dict, Any
import numpy as np
from .io import save_json, save_csv, save_embeddings


def export_results(metrics: Dict[str, np.ndarray], tc: float, meta_result: Dict[str, Any], 
                  concept: str, output_dir: str, comparison_metrics: Dict[str, float] = None, 
                  experiment_config: Dict[str, Any] = None) -> None:
    """
    Export all simulation results with anchor comparison
    
    Args:
        metrics: Dictionary containing simulation metrics arrays
        tc: Critical temperature value
        meta_result: Meta vector computation result
        concept: Concept name for file naming
        output_dir: Directory to save export files
        comparison_metrics: Optional anchor comparison metrics
        experiment_config: Optional experiment configuration
        
    Raises:
        IOError: If export fails
        KeyError: If required data is missing
    """
    # Validate required inputs
    if 'meta_vector' not in meta_result:
        raise KeyError("meta_result must contain 'meta_vector' key")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise IOError(f"Cannot create output directory {output_dir}: {e}")
    
    # Save metrics as CSV
    csv_filepath = os.path.join(output_dir, f"{concept}_metrics.csv")
    try:
        # Convert numpy arrays to lists for CSV export
        csv_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                   for k, v in metrics.items()}
        save_csv(csv_data, csv_filepath)
    except Exception as e:
        raise IOError(f"Failed to save metrics CSV: {e}")
    
    # Save summary as JSON
    summary = {
        'concept': concept,
        'critical_temperature': tc,
        'meta_vector_method': meta_result.get('method', 'unknown'),
        'simulation_params': {
            'temperature_range': [float(metrics['temperatures'][0]), 
                                float(metrics['temperatures'][-1])],
            'n_temperatures': len(metrics['temperatures'])
        }
    }
    
    # Add anchor comparison if available
    if comparison_metrics:
        summary['anchor_comparison'] = comparison_metrics
    
    # Add experiment configuration if available
    if experiment_config:
        summary['experiment_config'] = experiment_config
    
    summary_filepath = os.path.join(output_dir, f"{concept}_summary.json")
    try:
        save_json(summary, summary_filepath)
    except Exception as e:
        raise IOError(f"Failed to save summary JSON: {e}")
    
    # Save meta vector
    try:
        save_embeddings(meta_result['meta_vector'], f"{concept}_meta", "meta", output_dir)
    except Exception as e:
        raise IOError(f"Failed to save meta vector: {e}")
    
    # Save detailed comparison report if available
    if comparison_metrics:
        comparison_report = {
            'concept': concept,
            'critical_temperature': tc,
            'anchor_comparison_metrics': comparison_metrics,
            'experiment_config': experiment_config,
            'interpretation': {
                'procrustes_distance': 'Lower values indicate better structural alignment',
                'cka_similarity': 'Higher values (closer to 1.0) indicate stronger similarity',
                'emd_distance': 'Lower values indicate more similar distributions',
                'kl_divergence': 'Lower values indicate more similar probability distributions',
                'cosine_similarity': 'Higher values indicate more similar vector directions'
            }
        }
        
        comparison_filepath = os.path.join(output_dir, f"{concept}_anchor_comparison.json")
        try:
            save_json(comparison_report, comparison_filepath)
        except Exception as e:
            raise IOError(f"Failed to save comparison report: {e}") 