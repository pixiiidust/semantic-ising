import numpy as np
from scipy.stats import linregress
import logging
from .clustering import cluster_vectors

logger = logging.getLogger(__name__)


def find_critical_temperature(metrics_dict: dict, xi_threshold: float = 10) -> float:
    """
    Find critical temperature using combined logic:
    - Compute Tc_knee (log(xi) derivative, knee/inflection method)
    - Compute Tc_drop (first temperature where correlation length drops below xi_threshold)
    - If Tc_drop <= Tc_knee, use Tc_drop; else, use Tc_knee
    Falls back to old methods if correlation_length not present.
    """
    if 'correlation_length' in metrics_dict:
        correlation_length = metrics_dict['correlation_length']
        temperatures = metrics_dict['temperatures']
        if len(correlation_length) < 3:
            return np.nan
        # Only use finite, positive values
        mask = np.isfinite(correlation_length) & (correlation_length > 0)
        temps_valid = np.array(temperatures)[mask]
        xi_valid = np.array(correlation_length)[mask]
        if len(xi_valid) < 3:
            return np.nan
        # Knee detection (log(xi) derivative)
        dlogxi = np.gradient(np.log(xi_valid), temps_valid)
        tc_knee_idx = np.argmax(np.abs(dlogxi))
        tc_knee = float(temps_valid[tc_knee_idx])
        # First drop below threshold
        below = np.where(xi_valid < xi_threshold)[0]
        tc_drop = float(temps_valid[below[0]]) if len(below) > 0 else np.inf
        # Combined logic
        if tc_drop <= tc_knee:
            return tc_drop
        else:
            return tc_knee
    # Fallback to old methods if correlation_length not present
    if 'alignment' in metrics_dict:
        alignment = metrics_dict['alignment']
        temperatures = metrics_dict['temperatures']
        if len(alignment) < 3:
            return np.nan
        # Check for constant alignment (no transition)
        if np.allclose(alignment, alignment[0]):
            return np.nan
        # Try Binder cumulant method if ensemble data is available
        if 'alignment_ensemble' in metrics_dict and len(metrics_dict['alignment_ensemble']) > 0:
            try:
                return _find_critical_temperature_binder(metrics_dict)
            except Exception as e:
                logger.warning(f"Binder cumulant method failed: {e}, falling back to derivative method")
        # Fallback to derivative method
        return _find_critical_temperature_derivative(metrics_dict)
    return np.nan


def _find_critical_temperature_binder(metrics_dict: dict) -> float:
    """
    Find critical temperature using Binder cumulant method
    
    Args:
        metrics_dict: Dictionary containing 'temperatures' and 'alignment_ensemble' arrays
        
    Returns:
        Critical temperature value (Tc) or NaN if insufficient data
    """
    alignment_ensemble = metrics_dict['alignment_ensemble']
    temperatures = metrics_dict['temperatures']
    
    if len(alignment_ensemble) < 3:
        return np.nan
    
    # Convert list of arrays to 2D array (temperatures x sweeps)
    # Each row contains alignment values from different sweeps at that temperature
    ensemble_array = np.array(alignment_ensemble)
    
    # Compute Binder cumulant: U = 1 - <M^4>/(3*<M^2>^2)
    # For alignment, use U = 1 - <A^4>/(3*<A^2>^2)
    m2 = np.mean(ensemble_array**2, axis=1)  # <A^2> over sweeps
    m4 = np.mean(ensemble_array**4, axis=1)  # <A^4> over sweeps
    
    # Avoid division by zero
    valid_indices = m2 > 1e-10
    if not np.any(valid_indices):
        return np.nan
    
    binder_cumulant = np.full_like(m2, np.nan)
    binder_cumulant[valid_indices] = 1 - m4[valid_indices] / (3 * m2[valid_indices]**2)
    
    # Find minimum of Binder cumulant (critical point)
    # The Binder cumulant should have a minimum near the critical temperature
    valid_binder = binder_cumulant[valid_indices]
    valid_temps = temperatures[valid_indices]
    
    if len(valid_binder) < 2:
        return np.nan
    
    # Find the minimum of the Binder cumulant
    min_idx = np.argmin(valid_binder)
    return valid_temps[min_idx]


def _find_critical_temperature_derivative(metrics_dict: dict) -> float:
    """
    Find critical temperature using alignment derivative peak (fallback method)
    
    Args:
        metrics_dict: Dictionary containing 'temperatures' and 'alignment' arrays
        
    Returns:
        Critical temperature value (Tc) or NaN if insufficient data
    """
    alignment = metrics_dict['alignment']
    temperatures = metrics_dict['temperatures']
    
    if len(alignment) < 3:
        return np.nan
    
    # Compute derivative of alignment with respect to temperature
    # Use finite differences
    dM_dT = np.zeros_like(alignment)
    for i in range(1, len(alignment) - 1):
        dT = temperatures[i+1] - temperatures[i-1]
        if dT > 0:
            dM_dT[i] = (alignment[i+1] - alignment[i-1]) / dT
    
    # Set endpoints to nearest interior value
    dM_dT[0] = dM_dT[1]
    dM_dT[-1] = dM_dT[-2]
    
    # Find peak of absolute derivative (maximum rate of change)
    peak_idx = np.argmax(np.abs(dM_dT))
    
    return temperatures[peak_idx]


def detect_powerlaw_regime(vectors: np.ndarray, T: float, threshold: float = 0.8) -> dict:
    """
    Detect power law in cluster size distribution
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        T: Temperature at which clustering is performed
        threshold: Cosine similarity threshold for clustering
        
    Returns:
        Dictionary with 'exponent', 'r_squared', and 'n_clusters' keys
    """
    try:
        clusters = cluster_vectors(vectors, threshold)
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        if len(cluster_sizes) < 2:
            return {
                'exponent': np.nan,
                'r_squared': 0.0,
                'n_clusters': len(clusters)
            }
        
        # Count frequency of each cluster size
        size_counts = {}
        for size in cluster_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Prepare data for log-log fit
        sizes = list(size_counts.keys())
        counts = list(size_counts.values())
        
        # Require at least 3 distinct cluster sizes for meaningful power law fit
        if len(sizes) < 3:
            return {
                'exponent': np.nan,
                'r_squared': 0.0,
                'n_clusters': len(clusters)
            }
        
        # For 3 or more unique sizes, perform the fit
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        # Fit power law: P(s) ~ s^(-α)
        slope, intercept, r_value, _, _ = linregress(log_sizes, log_counts)
        
        return {
            'exponent': -slope,
            'r_squared': r_value**2,
            'n_clusters': len(clusters)
        }
        
    except Exception as e:
        logger.warning(f"Power law detection failed: {e}")
        return {
            'exponent': np.nan,
            'r_squared': 0.0,
            'n_clusters': 0
        } 