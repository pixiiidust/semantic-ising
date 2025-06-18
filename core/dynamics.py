import numpy as np
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)

# Minimum R² threshold for acceptable exponential fit
_MIN_R2 = 0.40


def compute_correlation_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix C_ij = cos(θ_ij) between normalized vectors
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        
    Returns:
        Correlation matrix of shape (n_vectors, n_vectors)
        
    Raises:
        ValueError: If vectors array is empty
    """
    if len(vectors) == 0:
        raise ValueError("Empty vectors array")
    
    n = len(vectors)
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Initialize correlation matrix
    C_matrix = np.zeros((n, n))
    
    # Compute pairwise correlations
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-correlation
                C_matrix[i, j] = np.dot(normalized_vectors[i], normalized_vectors[j])
    
    return C_matrix


def compute_correlation_length(vectors: np.ndarray, lang_dist_matrix: np.ndarray = None) -> float:
    """
    Compute correlation length from vector correlations with optional linguistic distance support
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        lang_dist_matrix: Optional distance matrix of shape (n_vectors, n_vectors)
                         for linguistic distance weighting
        
    Returns:
        Correlation length value (ξ) or NaN if insufficient data or poor fit
    """
    n = len(vectors)
    
    if n < 3:
        return np.nan
    
    # Compute correlation matrix
    C_matrix = compute_correlation_matrix(vectors)
    
    # Compute average correlation as function of distance
    correlations = []
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            correlations.append(abs(C_matrix[i, j]))
            
            # Use linguistic distance if provided, otherwise use index separation
            if lang_dist_matrix is not None:
                distances.append(lang_dist_matrix[i, j])
            else:
                distances.append(j - i)  # Default to index separation
    
    # Fit exponential decay: C(d) = C0 * exp(-d/ξ)
    if len(correlations) > 2:
        try:
            def exp_decay(x, C0, xi):
                return C0 * np.exp(-x/xi)
            
            popt, _ = curve_fit(exp_decay, distances, correlations, p0=[1.0, 1.0])
            C0_hat, xi_hat = popt
            
            # Compute R² to assess fit quality
            predictions = exp_decay(np.array(distances), *popt)
            ss_res = np.sum((np.array(correlations) - predictions) ** 2)
            ss_tot = np.sum((np.array(correlations) - np.mean(correlations)) ** 2)
            
            # Handle case where ss_tot is zero (identical vectors)
            if ss_tot == 0:
                # If all correlations are 1.0 (identical vectors), return large value
                if all(np.isclose(c, 1.0) for c in correlations):
                    return 1000.0
                # Otherwise, return NaN (e.g., all zeros)
                return np.nan
            
            r2 = 1 - ss_res / ss_tot
            
            # Return NaN if fit is poor or decay parameter is non-positive
            if r2 < _MIN_R2 or xi_hat <= 0:
                return np.nan
            
            return xi_hat  # ξ (correlation length)
        except RuntimeError:
            return np.nan
    
    return np.nan


def alignment_curvature(alignment_curve: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """
    Compute second derivative for Tc detection using finite differences
    
    Args:
        alignment_curve: Array of alignment values
        temperatures: Array of corresponding temperatures
        
    Returns:
        Curvature array (d²M/dT²) or empty array if insufficient points.
        Note: Endpoints are set to 0.0 by design (no neighbor on one side for finite difference)
    """
    if len(alignment_curve) < 3:
        return np.array([])
    
    d2M_dT2 = np.zeros_like(alignment_curve)
    
    for i in range(1, len(alignment_curve) - 1):
        dT = temperatures[i+1] - temperatures[i-1]
        if dT > 0:
            d2M_dT2[i] = (alignment_curve[i+1] - 2*alignment_curve[i] + alignment_curve[i-1]) / (dT**2)
    
    return d2M_dT2 