"""
Post-simulation analysis module for Semantic Ising Simulator.

This module provides comprehensive post-simulation analysis capabilities including:
- Simulation results analysis and interpretation
- Anchor language comparison at critical temperature
- Visualization data preparation for UI components
- Integration with power law and correlation analysis
- Results interpretation and insights generation
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Import existing core modules
from .comparison_metrics import compare_anchor_to_multilingual
from .simulation import compute_correlation_length, compute_correlation_matrix

# Set up logging
logger = logging.getLogger(__name__)


def _calculate_custom_anchor_metrics(anchor_vectors: np.ndarray, tc_vectors: np.ndarray, k: int = 3) -> Dict[str, float]:
    """
    Calculate custom anchor comparison metrics.

    Args:
        anchor_vectors: The anchor vector(s).
        tc_vectors: The vectors of other languages at the critical temperature.
        k: The number of nearest neighbors for the k-NN metric.

    Returns:
        A dictionary with the calculated metrics and the meta-vector.
    """
    if anchor_vectors is None or len(anchor_vectors) == 0:
        return {}
    if tc_vectors is None or len(tc_vectors) == 0:
        return {}
    
    # Assuming a single anchor vector for comparison
    anchor_vector = anchor_vectors[0]

    # 1. Calculate cosine similarity with the meta-vector
    meta_vector = np.mean(tc_vectors, axis=0)
    meta_vector /= np.linalg.norm(meta_vector)
    cos_anchor_meta_vector = np.dot(anchor_vector, meta_vector)

    # 2. Calculate average cosine similarity with k-Nearest Neighbors
    if len(tc_vectors) < k:
        logger.warning(f"Number of vectors ({len(tc_vectors)}) is less than k ({k}), using all vectors for k-NN.")
        k = len(tc_vectors)

    similarities = np.dot(tc_vectors, anchor_vector)
    top_k_indices = np.argsort(similarities)[-k:]
    avg_cos_anchor_knn = np.mean(similarities[top_k_indices])

    return {
        'metrics': {
            'cos_anchor_meta_vector': cos_anchor_meta_vector,
            'avg_cos_anchor_knn': avg_cos_anchor_knn,
            'cosine_similarity': cos_anchor_meta_vector, 
            'cosine_distance': 1 - cos_anchor_meta_vector
        },
        'meta_vector': meta_vector
    }


def analyze_simulation_results(
    simulation_results: Dict[str, Any], 
    anchor_vectors: np.ndarray, 
    tc: float,
    k_nn_value: int = 3
) -> Dict[str, Any]:
    """
    Perform post-simulation analysis including anchor comparison.
    
    This function performs comprehensive analysis of simulation results,
    including anchor language comparison at critical temperature,
    power law analysis, and correlation analysis. It handles cases
    where some temperatures may have been skipped due to divergence.
    
    Args:
        simulation_results: Dictionary containing simulation results
        anchor_vectors: Anchor language vectors for comparison
        tc: Critical temperature for analysis
        k_nn_value: The number of nearest neighbors for the k-NN metric
    
    Returns:
        Dictionary containing analysis results:
            - 'critical_temperature': Critical temperature value
            - 'anchor_comparison': Anchor comparison metrics
            - 'power_law_analysis': Power law analysis results
            - 'correlation_analysis': Correlation analysis results
    
    Raises:
        ValueError: If inputs are invalid or missing required data
    """
    # Input validation
    validate_analysis_inputs(simulation_results, anchor_vectors, tc)
    
    if anchor_vectors is None:
        raise ValueError("Anchor vectors cannot be None")
    
    if tc <= 0:
        raise ValueError("Critical temperature must be positive")
    
    # Extract metrics and validate structure
    # Handle both old format (metrics nested) and new format (direct)
    if 'metrics' in simulation_results:
        metrics = simulation_results['metrics']
    else:
        metrics = simulation_results
    
    if not metrics or 'temperatures' not in metrics:
        raise ValueError("Simulation results must contain valid metrics with temperatures")
    
    # Filter out any NaN values and ensure we have valid data
    temperatures = metrics['temperatures']
    alignment = metrics.get('alignment', [])
    
    # Create mask for valid data
    valid_mask = np.isfinite(temperatures)
    if 'alignment' in metrics:
        valid_mask &= np.isfinite(alignment)
    
    if not np.any(valid_mask):
        raise ValueError("All temperature points were invalid; check convergence or temperature range.")
    
    # Filter to only valid temperatures
    valid_temperatures = temperatures[valid_mask]
    valid_alignment = alignment[valid_mask] if len(alignment) > 0 else valid_temperatures
    
    # Find index closest to critical temperature among valid temperatures
    tc_idx = np.argmin(np.abs(valid_temperatures - tc))
    actual_tc = valid_temperatures[tc_idx]
    
    logger.info(f"Using temperature {actual_tc:.3f} (closest to Tc={tc:.3f}) for analysis")
    print(f"DEBUG: Post-analysis using temperature {actual_tc:.3f} (closest to Tc={tc:.3f})")
    
    # Extract vectors at critical temperature
    if 'vector_snapshots' in simulation_results and simulation_results['vector_snapshots']:
        # Find the closest available snapshot temperature to the critical temperature
        available_temps = list(simulation_results['vector_snapshots'].keys())
        closest_tc = min(available_temps, key=lambda t: abs(t - actual_tc))
        tc_vectors = simulation_results['vector_snapshots'][closest_tc]
        logger.info(f"Using vector snapshots at T = {closest_tc} (closest to {actual_tc})")
        print(f"DEBUG: Using vector snapshots at T = {closest_tc} (closest to {actual_tc})")
    else:
        # Fallback to original dynamics vectors
        tc_vectors = simulation_results.get('dynamics_vectors')
        if tc_vectors is None:
            # Try to get vectors from the first available snapshot
            if 'vector_snapshots' in simulation_results and simulation_results['vector_snapshots']:
                first_tc = list(simulation_results['vector_snapshots'].keys())[0]
                tc_vectors = simulation_results['vector_snapshots'][first_tc]
                logger.info(f"Using vectors from first available snapshot at T = {first_tc}")
                print(f"DEBUG: Using vectors from first available snapshot at T = {first_tc}")
            else:
                logger.error("No vectors available for analysis - no snapshots or dynamics_vectors found")
                print(f"DEBUG: No vectors available for analysis")
                # Return empty analysis results instead of raising error
                return {
                    'critical_temperature': tc,
                    'anchor_comparison': {
                        'cos_anchor_meta_vector': np.nan,
                        'avg_cos_anchor_knn': np.nan,
                        'cosine_similarity': np.nan,
                        'cosine_distance': np.nan
                    },
                    'correlation_analysis': {
                        'correlation_length': np.nan,
                        'correlation_matrix': np.array([])
                    }
                }
        else:
            logger.info(f"Using original dynamics vectors for analysis at T = {actual_tc}")
            print(f"DEBUG: Using original dynamics vectors for analysis at T = {actual_tc}")
    
    print(f"DEBUG: Available snapshot temperatures: {list(simulation_results.get('vector_snapshots', {}).keys())}")
    print(f"DEBUG: Actual Tc used for analysis: {actual_tc}")
    print(f"DEBUG: Closest snapshot temperature: {closest_tc if 'vector_snapshots' in simulation_results and simulation_results['vector_snapshots'] else 'N/A'}")
    
    # Perform anchor comparison at critical temperature
    try:
        comparison_results = _calculate_custom_anchor_metrics(
            anchor_vectors, 
            tc_vectors,
            k_nn_value
        )
        anchor_comparison = comparison_results.get('metrics', {})
        meta_vector_at_tc = comparison_results.get('meta_vector')
        logger.info("Custom anchor comparison completed successfully")
        print(f"DEBUG: Anchor comparison metrics calculated: {anchor_comparison}")
    except Exception as e:
        logger.error(f"Custom anchor comparison failed: {e}")
        print(f"DEBUG: Anchor comparison failed: {e}")
        anchor_comparison = {
            'cos_anchor_meta_vector': np.nan,
            'avg_cos_anchor_knn': np.nan,
            'cosine_similarity': np.nan,
            'cosine_distance': np.nan
        }
        meta_vector_at_tc = None
    
    # Perform correlation analysis at critical temperature
    try:
        correlation_length = compute_correlation_length(tc_vectors)
        correlation_matrix = compute_correlation_matrix(tc_vectors)
        correlation_analysis = {
            'correlation_length': correlation_length,
            'correlation_matrix': correlation_matrix
        }
        logger.info("Correlation analysis completed successfully")
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        correlation_analysis = {
            'correlation_length': np.nan,
            'correlation_matrix': np.array([])
        }
    
    # Compile comprehensive analysis results
    analysis_results = {
        'critical_temperature': actual_tc,
        'anchor_comparison': anchor_comparison,
        'correlation_analysis': correlation_analysis,
        'meta_vector_at_tc': meta_vector_at_tc
    }
    
    logger.info(f"Post-simulation analysis completed for T = {actual_tc}")
    return analysis_results


def generate_visualization_data(
    simulation_results: Dict[str, Any], 
    analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate data structures for UI visualizations and export.
    
    This function prepares all necessary data for visualization components,
    including temperature curves, anchor comparison metrics, power law
    analysis, and correlation data. It also handles vector evolution
    data when available.
    
    Args:
        simulation_results: Dictionary containing simulation results
        analysis_results: Dictionary containing post-analysis results
    
    Returns:
        Dictionary containing visualization-ready data:
            - 'temperature_curves': Temperature-dependent metrics for plotting
            - 'critical_temperature': Critical temperature for markers
            - 'anchor_comparison': Anchor comparison metrics for display
            - 'power_law': Power law analysis results
            - 'correlation_data': Correlation analysis results
            - 'vector_evolution': Optional vector snapshots for evolution plots
    
    Raises:
        ValueError: If inputs are invalid or missing required data
    """
    # Input validation
    if not simulation_results:
        raise ValueError("Simulation results cannot be empty")
    
    if not analysis_results:
        raise ValueError("Analysis results cannot be empty")
    
    # Extract metrics from simulation results (handle both old and new formats)
    if 'metrics' in simulation_results:
        # Old format: metrics nested under 'metrics' key
        metrics = simulation_results['metrics']
    else:
        # New format: metrics directly at top level
        metrics = simulation_results
    
    if not metrics:
        raise ValueError("Simulation results must contain metrics")
    
    # Prepare temperature curves data
    temperature_curves = {
        'temperatures': metrics.get('temperatures', np.array([])),
        'alignment': metrics.get('alignment', np.array([])),
        'entropy': metrics.get('entropy', np.array([])),
        'energy': metrics.get('energy', np.array([])),
        'correlation_length': metrics.get('correlation_length', np.array([]))
    }
    
    # Validate temperature curves data
    for key, value in temperature_curves.items():
        if not isinstance(value, np.ndarray) or len(value) == 0:
            logger.warning(f"Missing or invalid data for {key}")
            temperature_curves[key] = np.array([])
    
    # Extract analysis components
    critical_temperature = analysis_results.get('critical_temperature', np.nan)
    anchor_comparison = analysis_results.get('anchor_comparison', {})
    correlation_data = analysis_results.get('correlation_analysis', {})
    
    # Prepare visualization data structure
    viz_data = {
        'temperature_curves': temperature_curves,
        'critical_temperature': critical_temperature,
        'anchor_comparison': anchor_comparison,
        'correlation_data': correlation_data
    }
    
    # Add vector evolution data if available
    if 'vector_snapshots' in simulation_results:
        viz_data['vector_evolution'] = simulation_results['vector_snapshots']
        logger.info("Vector evolution data included in visualization data")
    else:
        logger.info("No vector snapshots available for evolution plots")
    
    # Validate visualization data structure
    required_keys = ['temperature_curves', 'critical_temperature', 'anchor_comparison', 'correlation_data']
    for key in required_keys:
        if key not in viz_data:
            logger.warning(f"Missing required key in visualization data: {key}")
    
    logger.info("Visualization data generation completed")
    return viz_data


def interpret_analysis_results(analysis_results: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate human-readable interpretations of analysis results.
    
    This function provides automatic interpretation of analysis results,
    including insights about anchor language similarity, power law
    characteristics, and correlation behavior.
    
    Args:
        analysis_results: Dictionary containing analysis results
    
    Returns:
        Dictionary containing interpretation strings for each analysis component
    """
    interpretations = {}
    
    # Interpret anchor comparison results
    anchor_comparison = analysis_results.get('anchor_comparison', {})
    cka_similarity = anchor_comparison.get('cosine_similarity', np.nan)
    
    if not np.isnan(cka_similarity):
        if cka_similarity > 0.7:
            interpretations['anchor_similarity'] = "Strong semantic similarity detected between anchor and multilingual structure"
        elif cka_similarity > 0.4:
            interpretations['anchor_similarity'] = "Moderate semantic similarity detected between anchor and multilingual structure"
        else:
            interpretations['anchor_similarity'] = "Weak semantic similarity detected between anchor and multilingual structure"
    else:
        interpretations['anchor_similarity'] = "Anchor similarity analysis unavailable"
    
    # Interpret correlation results
    correlation_data = analysis_results.get('correlation_analysis', {})
    correlation_length = correlation_data.get('correlation_length', np.nan)
    
    if not np.isnan(correlation_length):
        if correlation_length > 1.0:
            interpretations['correlation'] = f"Long-range correlations detected (ξ = {correlation_length:.2f})"
        else:
            interpretations['correlation'] = f"Short-range correlations detected (ξ = {correlation_length:.2f})"
    else:
        interpretations['correlation'] = "Correlation analysis unavailable"
    
    # Overall interpretation
    tc = analysis_results.get('critical_temperature', np.nan)
    if not np.isnan(tc):
        interpretations['overall'] = f"Critical temperature detected at T = {tc:.3f}. System shows phase transition behavior."
    else:
        interpretations['overall'] = "Critical temperature detection failed. Analysis may be incomplete."
    
    return interpretations


def validate_analysis_inputs(
    simulation_results: Dict[str, Any], 
    anchor_vectors: np.ndarray, 
    tc: float
) -> bool:
    """
    Validate inputs for post-simulation analysis.
    
    This function performs comprehensive validation of simulation results,
    anchor vectors, and critical temperature to ensure they are suitable
    for post-simulation analysis.
    
    Args:
        simulation_results: Dictionary containing simulation results
        anchor_vectors: Anchor language vectors for comparison
        tc: Critical temperature for analysis
    
    Returns:
        True if inputs are valid
    
    Raises:
        ValueError: If inputs are invalid or missing required data
    """
    # Basic input validation
    if not simulation_results:
        raise ValueError("Simulation results cannot be empty")
    
    if anchor_vectors is None:
        raise ValueError("Anchor vectors cannot be None")
    
    if tc <= 0:
        raise ValueError("Critical temperature must be positive")
    
    # Extract metrics and validate structure
    # Handle both old format (metrics nested) and new format (direct)
    if 'metrics' in simulation_results:
        metrics = simulation_results['metrics']
    else:
        metrics = simulation_results
    
    if not metrics:
        raise ValueError("Simulation results must contain metrics")
    
    # Check for required metrics
    required_metrics = ['temperatures', 'alignment']
    for metric in required_metrics:
        if metric not in metrics:
            raise ValueError(f"Missing required metric: {metric}")
        
    # Filter out NaN values and ensure we have valid data
    temperatures = metrics['temperatures']
    alignment = metrics['alignment']
    
    # Create mask for valid data
    valid_mask = np.isfinite(temperatures) & np.isfinite(alignment)
    
    if not np.any(valid_mask):
        raise ValueError(
            "All temperature points were diverging or invalid. "
            "Try lowering T_max or increasing MC steps."
        )
        
    # Check if we have enough valid points for analysis
    n_valid = np.sum(valid_mask)
    if n_valid < 3:
        raise ValueError(
            f"Only {n_valid} valid temperature points found. "
            "Need at least 3 points for meaningful analysis."
        )
    
    # Validate anchor vectors
    if anchor_vectors.size == 0:
        raise ValueError("Anchor vectors cannot be empty")
    
    if anchor_vectors.ndim != 2:
        raise ValueError("Anchor vectors must be 2D array")
    
    # Check if critical temperature is within valid range
    valid_temperatures = temperatures[valid_mask]
    if tc < np.min(valid_temperatures) or tc > np.max(valid_temperatures):
        logger.warning(
            f"Critical temperature {tc:.3f} is outside valid range "
            f"[{np.min(valid_temperatures):.3f}, {np.max(valid_temperatures):.3f}]"
        )
    
    return True


def clean_simulation_results(
    simulation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Clean simulation results by filtering out invalid (NaN) data.
    
    This function removes rows with NaN values from all metric arrays
    and corresponding convergence data, ensuring only valid data remains.
    
    Args:
        simulation_results: Dictionary containing simulation results
    
    Returns:
        Dictionary with cleaned simulation results (NaN values removed)
    
    Raises:
        ValueError: If no valid data remains after cleaning
    """
    # Extract metrics (handle both nested and direct formats)
    if 'metrics' in simulation_results:
        metrics = simulation_results['metrics']
    else:
        metrics = simulation_results
    
    # Get temperatures and alignment for validation
    temperatures = metrics.get('temperatures', np.array([]))
    alignment = metrics.get('alignment', np.array([]))
    
    if len(temperatures) == 0 or len(alignment) == 0:
        raise ValueError("No temperature or alignment data found")
    
    # Create mask for valid data - check all metric columns
    valid_mask = np.isfinite(temperatures) & np.isfinite(alignment)
    
    # Also check other metric columns if they exist
    for key, values in metrics.items():
        if (isinstance(values, np.ndarray) and 
            len(values) == len(temperatures) and 
            key not in ['temperatures', 'alignment']):
            valid_mask = valid_mask & np.isfinite(values)
    
    if not np.any(valid_mask):
        raise ValueError(
            "All temperature points were diverging or invalid. "
            "Try lowering T_max or increasing MC steps."
        )
    
    # Filter all metric arrays
    cleaned_metrics = {}
    for key, values in metrics.items():
        if isinstance(values, np.ndarray) and len(values) == len(temperatures):
            cleaned_metrics[key] = values[valid_mask]
        else:
            cleaned_metrics[key] = values
    
    # Filter convergence data if present
    if 'convergence_data' in simulation_results:
        cleaned_convergence_data = [
            simulation_results['convergence_data'][i] 
            for i in range(len(simulation_results['convergence_data'])) 
            if valid_mask[i]
        ]
    else:
        cleaned_convergence_data = []
    
    # Create cleaned results
    cleaned_results = {**simulation_results}
    if 'metrics' in simulation_results:
        cleaned_results['metrics'] = cleaned_metrics
    else:
        cleaned_results.update(cleaned_metrics)
    
    if cleaned_convergence_data:
        cleaned_results['convergence_data'] = cleaned_convergence_data
    
    return cleaned_results 