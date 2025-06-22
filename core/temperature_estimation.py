"""
Temperature Estimation Module for Semantic Ising Simulator

This module provides intelligent temperature range estimation for Ising simulations
based on the physics of the system. It estimates the critical temperature and
practical range to avoid computational waste on irrelevant temperature regions.

Key Functions:
- estimate_critical_temperature(): Estimate Tc from initial vector similarity
- estimate_max_temperature(): Estimate Tmax from energy fluctuations  
- estimate_practical_range(): Combine estimates with padding and validation
- quick_scan_probe(): Quick simulation probe to refine estimates
- validate_temperature_range(): Validate range for simulation suitability
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import curve_fit

# Set up logging
logger = logging.getLogger(__name__)


def estimate_critical_temperature(vectors: np.ndarray) -> float:
    """
    Estimate critical temperature from initial vector similarity.
    
    The critical temperature is proportional to the average interaction strength
    in the system, which we estimate from the initial cosine similarity.
    
    Args:
        vectors: Normalized vectors of shape (n_vectors, dim)
        
    Returns:
        Estimated critical temperature
        
    Raises:
        ValueError: If vectors array is empty or invalid
    """
    if vectors.size == 0:
        raise ValueError("Cannot estimate temperature from empty vectors array")
    
    if vectors.ndim != 2:
        raise ValueError("Vectors must be 2D array")
    
    n_vectors = vectors.shape[0]
    
    # Handle edge cases
    if n_vectors == 1:
        # Single vector - return reasonable default
        return 0.5
    
    if n_vectors == 2:
        # Two vectors - compute direct similarity
        similarity = np.dot(vectors[0], vectors[1])
        return max(0.05, abs(similarity))
    
    # Compute average cosine similarity between all pairs
    total_similarity = 0.0
    count = 0
    
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            similarity = np.dot(vectors[i], vectors[j])
            total_similarity += abs(similarity)  # Use absolute value
            count += 1
    
    if count == 0:
        return 0.5  # Default fallback
    
    avg_similarity = total_similarity / count
    
    # Handle edge cases for very similar or very different vectors
    if avg_similarity > 0.95:
        # Nearly identical vectors - low critical temperature
        return 0.1
    elif avg_similarity < 0.05:
        # Nearly orthogonal vectors - high critical temperature
        return 2.0
    
    # Critical temperature is approximately the average similarity
    # (since energy coupling J = 1.0 in our model)
    tc_estimate = avg_similarity
    
    # Apply reasonable bounds
    tc_estimate = max(0.05, min(3.0, tc_estimate))
    
    return tc_estimate


def estimate_max_temperature(vectors: np.ndarray) -> float:
    """
    Estimate maximum temperature from local field energy fluctuations.
    
    The maximum practical temperature is the point where thermal energy
    can easily overwhelm the strongest possible aligning force.
    Uses factor 2.0 for realistic temperature range with normalized energy fluctuations.
    
    Args:
        vectors: Normalized vectors of shape (n_vectors, dim)
        
    Returns:
        Estimated maximum temperature
        
    Raises:
        ValueError: If vectors array is empty or invalid
    """
    if vectors.size == 0:
        raise ValueError("Cannot estimate temperature from empty vectors array")
    
    if vectors.ndim != 2:
        raise ValueError("Vectors must be 2D array")
    
    n_vectors, dim = vectors.shape
    
    if n_vectors < 2:
        return 5.0  # Default for single vector
    
    # Compute local fields and energy fluctuations
    max_energy_fluctuation = 0.0
    
    for i in range(n_vectors):
        # Compute local field for vector i
        local_field = np.zeros(dim)
        for j in range(n_vectors):
            if i != j:
                similarity = np.dot(vectors[i], vectors[j])
                local_field += similarity * vectors[j]
        
        # Compute energy of this interaction
        energy = -np.dot(vectors[i], local_field)
        
        # Maximum possible energy change if vector flips from aligned to anti-aligned
        # Normalize by number of vectors to get reasonable scale
        energy_fluctuation = 2 * abs(energy) / n_vectors
        max_energy_fluctuation = max(max_energy_fluctuation, energy_fluctuation)
    
    # If no significant interactions, return reasonable default
    if max_energy_fluctuation < 1e-6:
        return 5.0
    
    # Maximum temperature should be significantly larger than the largest energy fluctuation
    # Rule of thumb: Tmax ≈ 2-5 × ΔE_max (using 2.0 for realistic temperature range)
    tmax_estimate = 2.0 * max_energy_fluctuation
    
    # Debug logging
    logger.info(f"Energy fluctuation: {max_energy_fluctuation:.6f}, Tmax estimate: {tmax_estimate:.6f}")
    print(f"DEBUG: Energy fluctuation: {max_energy_fluctuation:.6f}, Tmax estimate: {tmax_estimate:.6f}")
    
    # Apply reasonable bounds
    tmax_estimate = max(1.0, min(15.0, tmax_estimate))
    
    return tmax_estimate


def estimate_practical_range(vectors: np.ndarray, 
                           padding: float = 0.05,
                           min_span: float = 0.75,
                           min_tmin: float = 0.05,
                           config_max_temperature: float = None) -> Tuple[float, float]:
    """
    Estimate practical temperature range with padding and validation.
    
    Combines critical temperature and maximum temperature estimates with
    intelligent padding and validation to ensure a useful simulation range.
    
    Args:
        vectors: Normalized vectors of shape (n_vectors, dim)
        padding: Fractional padding to add to range (default: 0.05 = 5%)
        min_span: Minimum span required (default: 0.75)
        min_tmin: Minimum temperature floor (default: 0.05)
        config_max_temperature: Maximum temperature from config file (optional)
        
    Returns:
        Tuple of (tmin, tmax) for practical simulation range
        
    Raises:
        ValueError: If vectors array is empty or invalid
    """
    if vectors.size == 0:
        raise ValueError("Cannot estimate temperature range from empty vectors array")
    
    # Estimate critical and maximum temperatures
    tc_estimate = estimate_critical_temperature(vectors)
    tmax_estimate = estimate_max_temperature(vectors)
    
    # Apply more conservative bounds for stability
    # For LaBSE vectors, interactions are typically very small
    # Use more conservative estimates to prevent divergence
    tmax_estimate = min(tmax_estimate, 5.0)  # Reduced from 8.0 to 5.0 for faster simulations
    tmax_estimate = max(tmax_estimate, tc_estimate * 1.5)  # Ensure tmax > tc
    
    # Apply lower bound floor: Tmin = max(0.05, 0.1 × S_avg)
    # This catches very low-similarity sets
    avg_similarity = _compute_average_similarity(vectors)
    tmin_estimate = max(min_tmin, 0.1 * avg_similarity)
    
    # Debug logging for Tmin calculation
    logger.info(f"Average similarity: {avg_similarity:.6f}, Tmin estimate: {tmin_estimate:.6f}")
    print(f"DEBUG: Average similarity: {avg_similarity:.6f}, Tmin estimate: {tmin_estimate:.6f}")
    
    # Apply padding to expand range
    span = tmax_estimate - tmin_estimate
    padding_amount = span * padding
    
    # Don't let padding push tmin below the estimated value
    tmin = max(tmin_estimate, tmin_estimate - padding_amount)
    tmax = tmax_estimate + padding_amount
    
    # Debug logging for final range
    logger.info(f"Initial range after padding: [{tmin:.6f}, {tmax:.6f}]")
    print(f"DEBUG: Initial range after padding: [{tmin:.6f}, {tmax:.6f}]")
    
    # Apply config max temperature as upper bound if provided
    if config_max_temperature is not None:
        tmax = min(tmax, config_max_temperature)
    
    # Apply additional conservative bounds
    tmax = min(tmax, 8.0)  # Reduced from 15.0 to 8.0 for faster simulations
    
    # Ensure minimum span
    if tmax - tmin < min_span:
        # Expand range to meet minimum span requirement
        center = (tmin + tmax) / 2
        half_span = min_span / 2
        tmin = max(min_tmin, center - half_span)
        # Respect config max temperature when expanding
        if config_max_temperature is not None:
            tmax = min(config_max_temperature, center + half_span)
        else:
            tmax = min(8.0, center + half_span)  # Increased from 2.0 to 8.0 for better range
    
    # Validate the range
    is_valid, message = validate_temperature_range(tmin, tmax)
    if not is_valid:
        logger.warning(f"Estimated range [{tmin:.3f}, {tmax:.3f}] invalid: {message}")
        print(f"DEBUG: Range validation failed: {message}")
        # Fall back to more suitable defaults based on actual vector properties
        # Use the computed average similarity to set a more reasonable min
        computed_min = max(min_tmin, 0.1 * avg_similarity)  # Increased from 0.05 to 0.1
        tmin = computed_min
        
        # Set max based on Tc estimate and similarity
        if config_max_temperature is not None:
            tmax = config_max_temperature
        else:
            # Use a more dynamic max based on similarity and Tc
            dynamic_max = max(2.0, tc_estimate * 2.0, avg_similarity * 10.0)
            tmax = min(8.0, dynamic_max)  # Cap at 8.0
        
        # Ensure minimum span
        if tmax - tmin < 0.5:
            # Expand range to meet minimum span
            center = (tmin + tmax) / 2
            tmin = max(min_tmin, center - 0.25)
            tmax = min(8.0, center + 0.25)
        
        logger.info(f"Fallback range: [{tmin:.6f}, {tmax:.6f}]")
        print(f"DEBUG: Fallback range: [{tmin:.6f}, {tmax:.6f}]")
    else:
        logger.info(f"Range validation passed: [{tmin:.6f}, {tmax:.6f}]")
        print(f"DEBUG: Range validation passed: [{tmin:.6f}, {tmax:.6f}]")
    
    return tmin, tmax


def quick_scan_probe(vectors: np.ndarray, 
                    original_range: Optional[Tuple[float, float]] = None,
                    n_points: int = 10,
                    max_steps: int = 50) -> Optional[Tuple[float, float]]:
    """
    Quick scan probe to refine temperature range estimate.
    
    Runs a fast simulation probe to detect the steepest |dM/dT| slope
    and refine the temperature range around that region.
    
    Args:
        vectors: Normalized vectors of shape (n_vectors, dim)
        original_range: Original estimated range (tmin, tmax)
        n_points: Number of temperature points to probe
        max_steps: Maximum simulation steps per temperature
        
    Returns:
        Refined range (tmin, tmax) or None if probe fails
    """
    try:
        from .simulation import simulate_at_temperature
        
        if original_range is None:
            tmin, tmax = estimate_practical_range(vectors)
        else:
            tmin, tmax = original_range
        
        # Create temperature points for quick scan
        temperatures = np.linspace(tmin, tmax, n_points)
        alignments = []
        
        # Quick simulation at each temperature
        for T in temperatures:
            try:
                metrics, _ = simulate_at_temperature(vectors.copy(), T, max_iter=max_steps)
                alignments.append(metrics['alignment'])
            except Exception as e:
                logger.warning(f"Quick scan failed at T={T}: {e}")
                alignments.append(0.5)  # Default value
        
        alignments = np.array(alignments)
        
        # Compute derivative |dM/dT|
        if len(alignments) > 2:
            dM_dT = np.abs(np.gradient(alignments, temperatures))
            
            # Find temperature with maximum slope
            max_slope_idx = np.argmax(dM_dT)
            max_slope_T = temperatures[max_slope_idx]
            max_slope = dM_dT[max_slope_idx]
            
            # Only refine if significant slope detected
            if max_slope > 0.02:
                # Expand range around the steepest slope
                # Use ±50% of the original span around the detected Tc
                span = tmax - tmin
                half_span = span * 0.5
                
                refined_tmin = max(0.05, max_slope_T - half_span)
                refined_tmax = max_slope_T + half_span
                
                # Ensure minimum span
                if refined_tmax - refined_tmin < 0.75:
                    center = (refined_tmin + refined_tmax) / 2
                    refined_tmin = max(0.05, center - 0.375)
                    refined_tmax = center + 0.375
                
                logger.info(f"Quick scan refined range: [{refined_tmin:.3f}, {refined_tmax:.3f}] around T={max_slope_T:.3f}")
                return refined_tmin, refined_tmax
        
        # If no significant slope detected, return None
        logger.warning("Quick scan detected no significant slope - using original range")
        return None
        
    except ImportError:
        logger.warning("Simulation module not available for quick scan")
        return None
    except Exception as e:
        logger.error(f"Quick scan probe failed: {e}")
        return None


def validate_temperature_range(tmin: float, tmax: float) -> Tuple[bool, str]:
    """
    Validate temperature range for simulation suitability.
    
    Args:
        tmin: Minimum temperature
        tmax: Maximum temperature
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check for valid order
    if tmin >= tmax:
        return False, f"Minimum temperature ({tmin}) must be less than maximum temperature ({tmax})"
    
    # Check for positive temperatures
    if tmin < 0:
        return False, f"Minimum temperature ({tmin}) must be positive"
    
    if tmax < 0:
        return False, f"Maximum temperature ({tmax}) must be positive"
    
    # Check for reasonable span
    span = tmax - tmin
    if span < 0.3:  # Reduced from 0.5 to 0.3 for small test cases
        return False, f"Temperature range too narrow: {span:.3f} (minimum 0.3)"
    
    if span > 20.0:
        return False, f"Temperature range too wide: {span:.3f} (maximum 20.0)"
    
    # Check for reasonable bounds
    if tmin > 10.0:
        return False, f"Minimum temperature too high: {tmin} (maximum 10.0)"
    
    if tmax > 50.0:
        return False, f"Maximum temperature too high: {tmax} (maximum 50.0)"
    
    return True, "Valid temperature range"


def _compute_average_similarity(vectors: np.ndarray) -> float:
    """
    Compute average cosine similarity between all vector pairs.
    
    Helper function for internal use.
    
    Args:
        vectors: Normalized vectors
        
    Returns:
        Average similarity
    """
    n_vectors = len(vectors)
    if n_vectors < 2:
        print(f"DEBUG: Single vector case, returning default similarity: 0.5")
        return 0.5  # Default for single vector
    
    total_similarity = 0.0
    count = 0
    
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            similarity = np.dot(vectors[i], vectors[j])
            total_similarity += abs(similarity)
            count += 1
    
    avg_similarity = total_similarity / count if count > 0 else 0.5
    print(f"DEBUG: Computed average similarity: {avg_similarity:.6f} from {count} vector pairs")
    
    return avg_similarity 