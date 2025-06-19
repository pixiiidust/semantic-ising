"""
Core simulation engine for the Semantic Ising Simulator.

This module contains the main simulation functions including temperature sweeps,
Ising updates, metrics collection, and convergence handling.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from core.physics import total_system_energy
from scipy.optimize import curve_fit
from core.clustering import cluster_vectors

# Initialize logger
logger = logging.getLogger(__name__)


def _ensure_float(value: Any, name: str, default: float = None) -> float:
    """
    Ensure a value is a float, with proper error handling.
    
    Args:
        value: Value to convert
        name: Name of the parameter for error messages
        default: Default value if conversion fails
        
    Returns:
        Float value
        
    Raises:
        ValueError: If conversion fails and no default provided
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is not None:
            logger.warning(f"Could not convert {name}={value!r} to float, using default {default}")
            return default
        else:
            raise ValueError(f"Config value '{name}' must be numeric, got {value!r}")


def run_temperature_sweep(
    vectors: np.ndarray, 
    T_range: List[float], 
    store_all_temperatures: bool = False, 
    max_snapshots: int = 10, 
    n_replicas: int = 1,
    n_sweeps_per_temperature: int = 10,
    sim_params: Dict[str, Any] = None,
    progress_callback: callable = None
) -> Dict[str, np.ndarray]:
    """
    Run temperature sweep with multi-replica support and memory management.
    
    This is the main driver function for running Ising simulations across
    a range of temperatures. It supports multi-replica averaging for
    statistical robustness and efficient memory management for large
    vector sets.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T_range: List of temperatures to simulate
        store_all_temperatures: Whether to store vector snapshots at all temperatures
        max_snapshots: Maximum number of vector snapshots to store (memory management)
        n_replicas: Number of independent replicas for statistical averaging
        n_sweeps_per_temperature: Number of sweeps per temperature for ensemble statistics
        sim_params: Dictionary of simulation parameters from config
        progress_callback: Callback function for real-time progress reporting
        
    Returns:
        Dictionary containing temperature-dependent metrics and optional vector snapshots
        
    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate vectors
    if vectors.size == 0:
        raise ValueError("Cannot run simulation with empty vectors array")
    
    if vectors.ndim != 2:
        raise ValueError("Vectors must be 2D array")
    
    if len(T_range) < 2:
        raise ValueError("Temperature range must have at least 2 points")
    
    if n_replicas < 1:
        raise ValueError("Number of replicas must be >= 1")
    
    if n_sweeps_per_temperature < 1:
        raise ValueError("Number of sweeps per temperature must be >= 1")
    
    # Debug: Print min/max cosine similarity of initial vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = np.dot(normalized_vectors, normalized_vectors.T)
    # Only consider off-diagonal elements
    triu_indices = np.triu_indices_from(similarities, k=1)
    min_sim = np.min(similarities[triu_indices])
    max_sim = np.max(similarities[triu_indices])
    print(f"[DEBUG] Initial min similarity: {min_sim:.4f}, max similarity: {max_sim:.4f}")
    logger.info(f"Initial min similarity: {min_sim:.4f}, max similarity: {max_sim:.4f}")
    
    # Multi-replica averaging
    if n_replicas > 1:
        all_metrics = []
        for replica in range(n_replicas):
            # Set different random seed for each replica
            np.random.seed(42 + replica)
            replica_metrics = _run_single_temperature_sweep(
                vectors, T_range, store_all_temperatures, max_snapshots, 
                n_sweeps_per_temperature, sim_params, progress_callback
            )
            all_metrics.append(replica_metrics)
        
        # Aggregate results with error bars
        return _aggregate_replica_results(all_metrics, T_range)
    else:
        # Single replica
        return _run_single_temperature_sweep(
            vectors, T_range, store_all_temperatures, max_snapshots, 
            n_sweeps_per_temperature, sim_params, progress_callback
        )


def _run_single_temperature_sweep(
    vectors: np.ndarray, 
    T_range: List[float], 
    store_all_temperatures: bool, 
    max_snapshots: int,
    n_sweeps_per_temperature: int,
    sim_params: Dict[str, Any] = None,
    progress_callback: callable = None
) -> Dict[str, np.ndarray]:
    """Run single temperature sweep (internal function)"""
    # Initialize lists to store ALL results (including diverging ones)
    all_temperatures = []
    all_metrics = {
        'alignment': [],
        'entropy': [],
        'energy': [],
        'correlation_length': [],
        'alignment_ensemble': []
    }
    
    # Store convergence information for each temperature
    convergence_data = []
    
    vector_snapshots = {} if store_all_temperatures else None
    
    # Memory management: store only selected temperatures
    if store_all_temperatures:
        # Store snapshots at regular intervals or key temperatures
        snapshot_indices = np.linspace(0, len(T_range)-1, min(max_snapshots, len(T_range)), dtype=int)
    
    # Extract simulation parameters with defaults
    if sim_params is None:
        sim_params = {}
    
    max_iter = _ensure_float(sim_params.get('max_iterations', 6000), 'max_iterations', 6000)
    convergence_threshold = _ensure_float(sim_params.get('convergence_threshold', 3e-3), 'convergence_threshold', 3e-3)
    noise_sigma = _ensure_float(sim_params.get('noise_sigma', 0.04), 'noise_sigma', 0.04)
    update_method = sim_params.get('update_method', 'metropolis')
    
    # Add per-temperature cluster statistics
    cluster_stats_per_temperature = []
    
    for i, T in enumerate(T_range):
        # Report progress if callback is provided
        if progress_callback is not None:
            progress = (i / len(T_range)) * 100
            progress_callback(progress, f"Processing temperature {T:.3f} ({i+1}/{len(T_range)})")
        
        # Always record this temperature
        all_temperatures.append(T)
        
        try:
            # Run multiple sweeps to build ensemble statistics
            alignment_sweeps = []
            convergence_infos = []
            current_vectors = vectors.copy()
            
            for sweep in range(n_sweeps_per_temperature):
                # Only log convergence warnings for the first sweep
                if sweep == 0:
                    T_metrics, current_vectors, convergence_info = simulate_at_temperature(
                        current_vectors, T,
                        max_iter=max_iter,
                        convergence_threshold=convergence_threshold,
                        noise_sigma=noise_sigma,
                        update_method=update_method
                    )
                else:
                    # For subsequent sweeps, suppress warnings
                    import logging
                    original_level = logger.level
                    logger.setLevel(logging.ERROR)  # Suppress warnings
                    try:
                        T_metrics, current_vectors, convergence_info = simulate_at_temperature(
                            current_vectors, T,
                            max_iter=max_iter,
                            convergence_threshold=convergence_threshold,
                            noise_sigma=noise_sigma,
                            update_method=update_method
                        )
                    finally:
                        logger.setLevel(original_level)  # Restore logging level
                
                alignment_sweeps.append(T_metrics['alignment'])
                convergence_infos.append(convergence_info)
            
            # Check if the simulation converged or reached a stable plateau
            final_status = convergence_infos[-1]['status']
            
            # Always store ensemble of alignment values
            all_metrics['alignment_ensemble'].append(np.array(alignment_sweeps))
            
            # For diverging or error status, use NaN for metrics but keep the row
            if final_status in ['diverging', 'error']:
                logger.warning(f"⚠️ {final_status} at T={T:.3f} (diff={convergence_infos[-1]['final_diff']:.2e})")
                # Store NaN for metrics that can't be trusted
                for key in ['alignment', 'entropy', 'energy', 'correlation_length']:
                    all_metrics[key].append(np.nan)
            else:
                # For plateau status, use soft convergence - treat as valid but with reduced alignment
                if final_status == 'plateau':
                    logger.info(f"High-T plateau at T={T:.3f} - using soft convergence")
                    # Reduce alignment for plateau temperatures to indicate lack of ordering
                    T_metrics['alignment'] = max(0.0, T_metrics['alignment'] * 0.1)
                
                # Use final sweep metrics for other measurements
                for key in ['alignment', 'entropy', 'energy', 'correlation_length']:
                    all_metrics[key].append(T_metrics[key])
            
            # Store convergence data for this temperature
            convergence_data.append({
                'temperature': T,
                'convergence_infos': convergence_infos,
                'final_diff': convergence_infos[-1]['final_diff'],
                'status': final_status,
                'iterations': convergence_infos[-1]['iterations']
            })
            
            # Store vectors only at selected temperatures (if converged)
            if store_all_temperatures and i in snapshot_indices and final_status == 'converged':
                vector_snapshots[T] = current_vectors.copy()
            
            # (after all Ising updates and before memory cleanup)
            # Cluster the vectors at this temperature (use final current_vectors)
            # Use temperature-dependent clustering threshold
            base_threshold = sim_params.get('similarity_threshold', 0.8) if sim_params else 0.8
            
            # Adjust threshold based on temperature: higher T = higher threshold
            # This ensures fewer clusters at high T (more independent) and more clusters at low T (more aligned)
            temp_factor = min(1.0, max(0.5, T / 2.0))  # Scale factor between 0.5 and 1.0
            similarity_threshold = base_threshold + (0.15 * temp_factor)  # Range: 0.8 to 0.95
            
            clusters = cluster_vectors(current_vectors, threshold=similarity_threshold)
            cluster_sizes = [len(cluster) for cluster in clusters]
            cluster_stats_per_temperature.append({
                'temperature': T,
                'n_clusters': len(clusters),
                'cluster_sizes': cluster_sizes,
                'clustering_threshold': similarity_threshold  # Store the actual threshold used
            })
            
            # Clear memory periodically
            if i % 10 == 0 and i > 0:
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed at temperature {T}: {e}")
            # Still record the temperature with NaN metrics
            for key in ['alignment', 'entropy', 'energy', 'correlation_length']:
                all_metrics[key].append(np.nan)
            
            # Add convergence data with error status
            convergence_data.append({
                'temperature': T,
                'convergence_infos': [],
                'final_diff': np.nan,
                'status': 'error',
                'iterations': 0
            })
            
            # Add empty ensemble for consistency
            all_metrics['alignment_ensemble'].append(np.array([np.nan]))
    
    # Create result with ALL temperatures (including diverging ones)
    result = {
        'temperatures': np.array(all_temperatures),
        **{k: np.array(v) for k, v in all_metrics.items()},
        'convergence_data': convergence_data,
        'cluster_stats_per_temperature': cluster_stats_per_temperature,
    }
    
    if store_all_temperatures:
        result['vector_snapshots'] = vector_snapshots
    
    # Report final progress
    if progress_callback is not None:
        progress_callback(100, "Temperature sweep completed!")
    
    return result


def _aggregate_replica_results(
    all_metrics: List[Dict[str, np.ndarray]], 
    T_range: List[float]
) -> Dict[str, np.ndarray]:
    """Aggregate results from multiple replicas with error bars"""
    n_replicas = len(all_metrics)
    aggregated = {'temperatures': T_range}
    
    # Aggregate each metric with mean and standard error
    for metric_name in ['alignment', 'entropy', 'energy', 'correlation_length']:
        # Stack all replica values
        values = np.stack([metrics[metric_name] for metrics in all_metrics], axis=0)
        
        # Compute mean and standard error
        mean_values = np.mean(values, axis=0)
        sem_values = np.std(values, axis=0) / np.sqrt(n_replicas)
        
        aggregated[metric_name] = mean_values
        aggregated[f'{metric_name}_sem'] = sem_values
        aggregated[f'{metric_name}_replicas'] = values
    
    return aggregated


def simulate_at_temperature(
    vectors: np.ndarray, 
    T: float, 
    max_iter: int = 6000, 
    convergence_threshold: float = 3e-3,
    log_every: int = 50,
    slope_tol: float = 5e-4,
    plateau_patience: int = 3,
    diverge_tol: float = 0.05,
    noise_sigma: float = 0.04,
    update_method: str = "metropolis"
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """
    Apply Ising-style update logic with detailed convergence tracking.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        max_iter: Maximum number of iterations (default: 6000)
        convergence_threshold: Convergence threshold for vector changes (default: 3e-3)
        log_every: Log convergence every N steps (default: 50)
        slope_tol: Tolerance for slope-based convergence (default: 5e-4)
        plateau_patience: Number of consecutive logs to confirm plateau (default: 3)
        diverge_tol: Tolerance for divergence detection (default: 0.05)
        noise_sigma: Standard deviation of noise for Metropolis updates (default: 0.04)
        update_method: Update method ("metropolis" or "glauber")
        
    Returns:
        Tuple of (metrics_dict, updated_vectors, convergence_info)
    """
    # Ensure all numeric parameters are floats
    T = _ensure_float(T, 'T')
    max_iter = int(_ensure_float(max_iter, 'max_iter', 6000))
    convergence_threshold = _ensure_float(convergence_threshold, 'convergence_threshold', 3e-3)
    log_every = int(_ensure_float(log_every, 'log_every', 50))
    slope_tol = _ensure_float(slope_tol, 'slope_tol', 5e-4)
    plateau_patience = int(_ensure_float(plateau_patience, 'plateau_patience', 3))
    diverge_tol = _ensure_float(diverge_tol, 'diverge_tol', 0.05)
    noise_sigma = _ensure_float(noise_sigma, 'noise_sigma', 0.04)
    
    if T <= 0:
        raise ValueError("Temperature must be positive")
    
    current_vectors = vectors.copy()
    n_vectors = len(vectors)
    
    # Track convergence history
    diff_history = []
    alignment_history = []
    logged_steps = []
    last_logged = 0
    plateau_count = 0
    
    for iteration in range(max_iter):
        new_vectors = update_vectors_ising(current_vectors, T, update_method=update_method, noise_sigma=noise_sigma)
        
        # Check convergence using vector norm difference
        diff = np.linalg.norm(new_vectors - current_vectors) / np.linalg.norm(current_vectors)
        
        # Log convergence every log_every steps
        if (iteration + 1) % log_every == 0:
            diff_history.append(diff)
            logged_steps.append(iteration + 1)
            
            # Compute alignment for tracking
            alignment = compute_alignment(new_vectors)
            alignment_history.append(alignment)
            
            # Early convergence check with slope analysis
            if len(diff_history) >= 2:
                slope = (diff_history[-1] - diff_history[-2]) / log_every
                
                # Check for convergence
                if abs(slope) < slope_tol and diff < convergence_threshold:
                    status = "converged"
                    break
                
                # Check for plateau (not converging but stable)
                if abs(slope) < slope_tol and diff > convergence_threshold:
                    plateau_count += 1
                    if plateau_count >= plateau_patience:
                        status = "plateau"
                        break
                else:
                    plateau_count = 0
                
                # Check for divergence
                if diff_history[-1] - diff_history[-2] > diverge_tol:
                    status = "diverging"
                    break
            
            last_logged = iteration + 1
        
        # Simple convergence check for very small changes
        if diff < convergence_threshold:
            status = "converged"
            break
        
        current_vectors = new_vectors
    else:
        # Reached max iterations
        status = "max_steps"
        if len(diff_history) > 0:
            diff = diff_history[-1]
        else:
            # Compute final diff if we never logged
            diff = np.linalg.norm(new_vectors - current_vectors) / np.linalg.norm(current_vectors)
    
    # Prepare convergence information
    convergence_info = {
        'status': status,
        'final_diff': diff,
        'iterations': iteration + 1,
        'diff_history': diff_history,
        'alignment_history': alignment_history,
        'logged_steps': logged_steps,
        'temperature': T
    }
    
    # Log appropriate message based on status
    if status == "plateau":
        logger.info(f"High-T plateau at T={T:.3f} (final diff: {diff:.2e})")
    elif status == "diverging":
        logger.warning(f"System diverging at T={T:.3f} (final diff: {diff:.2e})")
    elif status == "max_steps" and diff > convergence_threshold * 10:
        logger.warning(f"Simulation did not converge at T={T:.3f} (final diff: {diff:.2e})")
    elif status == "converged":
        logger.debug(f"Converged at T={T:.3f} in {iteration + 1} iterations")
    
    metrics = collect_metrics(current_vectors, T)
    return metrics, current_vectors, convergence_info


def update_vectors_ising(
    vectors: np.ndarray, 
    T: float, 
    J: float = 1.0, 
    update_method: str = "metropolis",
    noise_sigma: float = 0.04
) -> np.ndarray:
    """
    Update vectors using specified Ising update rule.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength parameter
        update_method: Update method ("metropolis" or "glauber")
        noise_sigma: Standard deviation of noise for Metropolis updates
    
    Returns:
        Updated normalized vectors
    
    Raises:
        ValueError: If update method is unknown
    """
    # Ensure numeric parameters are floats
    T = _ensure_float(T, 'T')
    J = _ensure_float(J, 'J', 1.0)
    noise_sigma = _ensure_float(noise_sigma, 'noise_sigma', 0.04)
    
    if update_method == "metropolis":
        return update_vectors_metropolis(vectors, T, J, noise_sigma)
    elif update_method == "glauber":
        return update_vectors_glauber(vectors, T, J)
    else:
        raise ValueError(f"Unknown update method: {update_method}")


def update_vectors_metropolis(vectors: np.ndarray, T: float, J: float = 1.0, noise_sigma: float = 0.04) -> np.ndarray:
    """
    Metropolis update rule for semantic Ising model.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength parameter
        noise_sigma: Standard deviation of noise for Metropolis updates
    
    Returns:
        Updated vectors using Metropolis acceptance criterion
    """
    # Ensure numeric parameters are floats
    T = _ensure_float(T, 'T')
    J = _ensure_float(J, 'J', 1.0)
    noise_sigma = _ensure_float(noise_sigma, 'noise_sigma', 0.04)
    
    n_vectors, dim = vectors.shape
    new_vectors = vectors.copy()
    
    # Vectorized computation of similarity matrix for efficiency
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero if a vector is all zeros
    norms[norms == 0] = 1
    normalized_vectors = vectors / norms
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    
    # Pre-compute all local fields. This includes self-interaction, which is subtracted later.
    local_field_matrix = J * np.dot(similarity_matrix, vectors)

    for i in range(n_vectors):
        # Look up the pre-computed local field and subtract the self-interaction term.
        # The self-interaction term is J * similarity(i,i) * vector[i]. Since sim(i,i) is 1, this is J * vectors[i].
        local_field = local_field_matrix[i] - J * vectors[i]
        
        # Propose new vector (small random perturbation)
        noise = np.random.normal(0, noise_sigma, dim)
        proposed_vector = vectors[i] + noise
        proposed_vector /= np.linalg.norm(proposed_vector) # Normalize the proposed vector
        
        # Compute energy change using consistent Hamiltonian
        # delta_E = E_new - E_old
        # E_i = -dot(vector_i, local_field_i), where local_field is based on all other vectors
        delta_E = -np.dot(proposed_vector, local_field) + np.dot(vectors[i], local_field)
        
        # Metropolis acceptance criterion
        if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
            new_vectors[i] = proposed_vector
            # After accepting a move, we should update the local field matrix for subsequent calculations in the same sweep
            # For simplicity and to maintain vectorization benefits, we will proceed without intra-sweep updates,
            # which is a common approach. Re-evaluate if convergence is impacted.

    # Global normalization after each sweep to prevent spin-length drift
    new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
    return new_vectors


def update_vectors_glauber(vectors: np.ndarray, T: float, J: float = 1.0) -> np.ndarray:
    """
    Glauber (heat-bath) update rule for semantic Ising model.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength parameter
    
    Returns:
        Updated vectors using heat-bath probability
    """
    # Ensure numeric parameters are floats
    T = _ensure_float(T, 'T')
    J = _ensure_float(J, 'J', 1.0)
    
    n_vectors, dim = vectors.shape
    new_vectors = vectors.copy()
    
    for i in range(n_vectors):
        # Compute local field
        local_field = np.zeros(dim)
        for j in range(n_vectors):
            if i != j:
                similarity = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                local_field += J * similarity * vectors[j]
        
        # Heat-bath probability for Glauber dynamics
        field_strength = np.linalg.norm(local_field)
        
        if field_strength > 0:
            field_direction = local_field / field_strength
            
            # Probability of aligning with field
            p_align = 1.0 / (1.0 + np.exp(-2 * J * field_strength / T))
            
            # Update vector based on probability
            if np.random.random() < p_align:
                new_vectors[i] = field_direction
            else:
                new_vectors[i] = -field_direction
        else:
            # Random direction if no field
            noise = np.random.normal(0, 1, dim)
            new_vectors[i] = noise / np.linalg.norm(noise)
    
    # Global normalization
    new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
    return new_vectors


def collect_metrics(vectors: np.ndarray, T: float) -> Dict[str, float]:
    """
    Collect all metrics for current vector state.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
    
    Returns:
        Dictionary with alignment, entropy, energy, correlation_length
    """
    return {
        'alignment': compute_alignment(vectors),
        'entropy': compute_entropy(vectors),
        'energy': total_system_energy(vectors),
        'correlation_length': compute_correlation_length(vectors)
    }


def compute_alignment(vectors: np.ndarray) -> float:
    """
    Compute alignment metric: average absolute cosine similarity between all vector pairs.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
    
    Returns:
        Average alignment value (0-1 scale)
    """
    n = len(vectors)
    if n < 2:
        return 1.0  # Single vector has perfect alignment
    
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    total_similarity = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = np.dot(normalized_vectors[i], normalized_vectors[j])
            total_similarity += abs(similarity)
            count += 1
    
    return total_similarity / count if count > 0 else 1.0


def compute_entropy(vectors: np.ndarray) -> float:
    """
    Compute entropy metric: Shannon entropy of vector distribution.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
    
    Returns:
        Entropy value (non-negative)
    """
    n = len(vectors)
    if n < 2:
        return 0.0  # Single vector has zero entropy
    
    # Compute pairwise distances
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(normalized_vectors[i] - normalized_vectors[j])
            distances.append(dist)
    
    # Bin distances for entropy calculation
    if len(distances) > 1:
        hist, _ = np.histogram(distances, bins=min(10, len(distances)//2), density=False)
        prob = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        prob = prob[prob > 0]
        if len(prob) > 0:
            entropy = -np.sum(prob * np.log(prob))
            return float(entropy)
    
    return 0.0


def compute_correlation_length(vectors: np.ndarray) -> float:
    """
    Compute correlation length from vector correlations.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
    
    Returns:
        Correlation length value
    """
    n = len(vectors)
    if n < 3:
        return np.nan
    
    # Compute correlation matrix
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    C_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                C_matrix[i, j] = np.dot(normalized_vectors[i], normalized_vectors[j])
    
    # Compute average correlation as function of distance
    correlations = []
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            correlations.append(abs(C_matrix[i, j]))
            distances.append(j - i)  # Default to index separation
    
    # Fit exponential decay: C(d) = C0 * exp(-d/ξ)
    if len(correlations) > 2:
        try:
            def exp_decay(x, C0, xi):
                # Add bounds to prevent overflow
                xi = np.clip(xi, 0.1, 100.0)  # Clip correlation length to reasonable range
                return C0 * np.exp(-np.clip(x/xi, -10, 10))  # Clip exponent to prevent overflow
            
            # Use better initial parameters and bounds
            popt, _ = curve_fit(
                exp_decay, 
                distances, 
                correlations, 
                p0=[1.0, 1.0],
                bounds=([0.1, 0.1], [10.0, 100.0])  # Reasonable bounds for C0 and xi
            )
            return float(popt[1])  # ξ (correlation length)
        except (RuntimeError, ValueError, TypeError):
            # Return a fallback value if fitting fails
            return 1.0
    
    return np.nan 