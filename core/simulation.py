"""
Core simulation engine for the Semantic Ising Simulator.

This module contains the main simulation functions including temperature sweeps,
Ising updates, metrics collection, and convergence handling.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from core.physics import total_system_energy
from scipy.optimize import curve_fit
from core.clustering import cluster_vectors
import os
import hashlib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob

# Initialize logger
logger = logging.getLogger(__name__)


def _get_snapshot_directory(concept: str, encoder: str, anchor_language: str, include_anchor: bool) -> str:
    """
    Generate a unique snapshot directory path based on simulation parameters.
    
    Args:
        concept: The concept being simulated
        encoder: The embedding model used
        anchor_language: The anchor language
        include_anchor: Whether anchor is included in dynamics
        
    Returns:
        Path to the snapshot directory
    """
    # Create a hash of the simulation parameters for uniqueness
    params_str = f"{concept}_{encoder}_{anchor_language}_{include_anchor}"
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Create snapshot directory
    snapshot_dir = os.path.join("data", "snapshots", f"{concept}_{params_hash}")
    os.makedirs(snapshot_dir, exist_ok=True)
    
    return snapshot_dir


def _save_snapshot_to_disk(snapshot_dir: str, temperature: float, vectors: np.ndarray, 
                          languages: List[str], metadata: Dict[str, Any]) -> None:
    """
    Save a vector snapshot to disk using NumPy-compatible serialization.
    
    Args:
        snapshot_dir: Directory to save the snapshot
        temperature: Temperature of the snapshot
        vectors: Vector data to save
        languages: List of language codes
        metadata: Additional metadata to save
    """
    try:
        # Create filename with temperature
        base_filename = f"snapshot_T{temperature:.6f}"
        pkl_filepath = os.path.join(snapshot_dir, f"{base_filename}.pkl")
        npy_filepath = os.path.join(snapshot_dir, f"{base_filename}_vectors.npy")
        
        # Save vectors using numpy.save (NumPy-compatible)
        np.save(npy_filepath, vectors)
        
        # Save metadata using pickle (without vectors)
        snapshot_metadata = {
            'temperature': temperature,
            'languages': languages,
            'metadata': metadata,
            'vectors_file': f"{base_filename}_vectors.npy"  # Reference to vectors file
        }
        
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(snapshot_metadata, f)
        
        logger.info(f"Saved snapshot to {pkl_filepath} and vectors to {npy_filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save snapshot at T={temperature}: {e}")


def _load_snapshot_from_disk(snapshot_dir: str, temperature: float) -> Optional[Dict[str, Any]]:
    """
    Load snapshot from disk with NumPy version compatibility handling.
    
    Args:
        snapshot_dir: Directory containing snapshots
        temperature: Temperature to load
        
    Returns:
        Snapshot data or None if loading fails
    """
    try:
        # Find the closest temperature file
        snapshot_files = glob.glob(os.path.join(snapshot_dir, "snapshot_T*.pkl"))
        if not snapshot_files:
            logger.warning(f"No snapshot files found in {snapshot_dir}")
            return None
        
        # Extract temperatures from filenames
        temp_files = []
        for file in snapshot_files:
            try:
                # Extract temperature from filename like "snapshot_T1.234.pkl"
                temp_str = os.path.basename(file).replace("snapshot_T", "").replace(".pkl", "")
                temp_val = float(temp_str)
                temp_files.append((temp_val, file))
            except ValueError:
                continue
        
        if not temp_files:
            logger.warning(f"No valid temperature files found in {snapshot_dir}")
            return None
        
        # Find closest temperature
        closest_temp, closest_file = min(temp_files, key=lambda x: abs(x[0] - temperature))
        
        # Try to load the snapshot with NumPy-compatible method
        try:
            with open(closest_file, 'rb') as f:
                # Load metadata from pickle
                snapshot_data = pickle.load(f)
                
                # Load vectors from separate .npy file
                if isinstance(snapshot_data, dict) and 'vectors_file' in snapshot_data:
                    vectors_file = os.path.join(snapshot_dir, snapshot_data['vectors_file'])
                    if os.path.exists(vectors_file):
                        vectors = np.load(vectors_file)
                        snapshot_data['vectors'] = vectors
                        logger.info(f"Successfully loaded snapshot from {closest_file} with vectors from {vectors_file}")
                    else:
                        logger.warning(f"Vectors file {vectors_file} not found")
                        return None
                else:
                    # Fallback: try to load old format
                    logger.warning(f"Old snapshot format detected in {closest_file}")
                    return None
            
            # Validate snapshot data
            if isinstance(snapshot_data, dict) and 'vectors' in snapshot_data and 'languages' in snapshot_data:
                return snapshot_data
            else:
                logger.warning(f"Invalid snapshot format in {closest_file}")
                return None
                
        except (ModuleNotFoundError, ImportError) as e:
            if "numpy._core" in str(e):
                logger.error(f"NumPy version incompatibility when loading snapshot at T={closest_temp}: {e}")
                logger.error("This indicates snapshots were saved with a different NumPy version")
                logger.error("Please re-run the simulation to generate compatible snapshots")
                return None
            else:
                logger.error(f"Import error loading snapshot: {e}")
                return None
        except Exception as e:
            logger.error(f"Error loading snapshot from {closest_file}: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error in _load_snapshot_from_disk: {e}")
        return None


def _get_available_snapshot_temperatures(snapshot_dir: str) -> List[float]:
    """
    Get list of available snapshot temperatures from disk.
    
    Args:
        snapshot_dir: Directory containing snapshots
        
    Returns:
        List of available temperatures
    """
    try:
        if not os.path.exists(snapshot_dir):
            return []
            
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith("snapshot_T") and f.endswith(".pkl")]
        
        available_temps = []
        for filename in snapshot_files:
            try:
                temp_str = filename.replace("snapshot_T", "").replace(".pkl", "")
                temp = float(temp_str)
                available_temps.append(temp)
            except ValueError:
                continue
        
        return sorted(available_temps)
        
    except Exception as e:
        logger.error(f"Failed to get available snapshot temperatures: {e}")
        return []


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
    progress_callback: callable = None,
    snapshot_dir: str = None,
    concept: str = None,
    encoder: str = None,
    anchor_language: str = None,
    include_anchor: bool = None,
    languages: List[str] = None
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
        snapshot_dir: Directory to save snapshots
        concept: The concept being simulated
        encoder: The embedding model used
        anchor_language: The anchor language
        include_anchor: Whether anchor is included in dynamics
        languages: List of language codes
        
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
                n_sweeps_per_temperature, sim_params, progress_callback,
                snapshot_dir, concept, encoder, anchor_language, include_anchor, languages
            )
            all_metrics.append(replica_metrics)
        
        # Aggregate results with error bars
        return _aggregate_replica_results(all_metrics, T_range)
    else:
        # Single replica
        return _run_single_temperature_sweep(
            vectors, T_range, store_all_temperatures, max_snapshots, 
            n_sweeps_per_temperature, sim_params, progress_callback,
            snapshot_dir, concept, encoder, anchor_language, include_anchor, languages
        )


def _run_single_temperature_sweep(
    vectors: np.ndarray, 
    T_range: List[float], 
    store_all_temperatures: bool, 
    max_snapshots: int,
    n_sweeps_per_temperature: int,
    sim_params: Dict[str, Any] = None,
    progress_callback: callable = None,
    snapshot_dir: str = None,
    concept: str = None,
    encoder: str = None,
    anchor_language: str = None,
    include_anchor: bool = None,
    languages: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Run a single temperature sweep with k-NN constraints.
    
    Args:
        vectors: Initial vectors array
        T_range: List of temperatures to simulate
        store_all_temperatures: Whether to store vectors at all temperatures
        max_snapshots: Maximum number of snapshots to store
        n_sweeps_per_temperature: Number of sweeps per temperature
        sim_params: Simulation parameters
        progress_callback: Progress callback function
        snapshot_dir: Directory for disk-based snapshot storage
        concept: Concept name for metadata
        encoder: Encoder name for metadata
        anchor_language: Anchor language code
        include_anchor: Whether anchor is included in dynamics
        languages: List of language codes
        
    Returns:
        Dictionary containing simulation results
    """
    # Ensure vectors are normalized
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Create diagnostic similarity matrix
    if languages:
        heatmap_path = create_similarity_matrix_heatmap(vectors, languages)
        if heatmap_path:
            logger.info(f"Diagnostic similarity matrix created: {heatmap_path}")
    
    # Get k-NN parameter from sim_params
    k = sim_params.get('k_neighbors', 8) if sim_params else 8
    logger.info(f"Using k-NN constraint with k={k}")
    
    # Initialize tracking variables
    current_vectors = vectors.copy()
    all_temperatures = []
    all_metrics = {
        'alignment': [],
        'entropy': [],
        'energy': [],
        'correlation_length': [],
        'alignment_ensemble': []
    }
    convergence_data = []
    cluster_stats_per_temperature = []
    
    # Determine snapshot indices
    if store_all_temperatures:
        n_steps = len(T_range)
        if n_steps <= max_snapshots:
            snapshot_indices = np.arange(n_steps)
        else:
            # Sample evenly across the range
            snapshot_indices = np.linspace(0, n_steps - 1, max_snapshots, dtype=int)
        
        logger.info(f"store_all_temperatures=True, max_snapshots={max_snapshots}")
        logger.info(f"snapshot_indices={snapshot_indices}")
        logger.info(f"snapshot_dir={snapshot_dir}")
    
    # Initialize vector snapshots for memory storage (fallback)
    vector_snapshots = {}
    
    # Run temperature sweep
    for i, T in enumerate(T_range):
        try:
            all_temperatures.append(T)
            
            # Update progress
            if progress_callback is not None:
                progress_percent = (i / len(T_range)) * 100
                progress_callback(progress_percent, f"Simulating at T={T:.3f}")
            
            # Run multiple sweeps at this temperature
            convergence_infos = []
            initial_vectors = current_vectors.copy()  # Store initial state
            
            for sweep in range(n_sweeps_per_temperature):
                # Apply k-NN constrained Ising updates
                current_vectors = update_vectors_ising(
                    current_vectors, 
                    T, 
                    update_method=sim_params.get('update_method', 'metropolis') if sim_params else 'metropolis',
                    noise_sigma=sim_params.get('noise_sigma', 0.04) if sim_params else 0.04,
                    k=k  # Use k-NN constraint
                )
                
                # Compute convergence info for this sweep
                if sweep == n_sweeps_per_temperature - 1:  # Only on final sweep
                    # Calculate actual final difference from initial state
                    final_diff = np.linalg.norm(current_vectors - initial_vectors) / np.linalg.norm(initial_vectors)
                    
                    convergence_info = {
                        'sweep': sweep + 1,
                        'temperature': T,
                        'final_diff': final_diff,  # Actual computed difference
                        'iterations': sweep + 1,
                        'status': 'completed'
                    }
                    convergence_infos.append(convergence_info)
            
            # Compute final metrics for this temperature
            T_metrics = collect_metrics(current_vectors, T)
            
            # Determine convergence status
            final_status = 'plateau'  # Default status for k-NN constrained dynamics
            
            # Store metrics
            all_metrics['alignment'].append(T_metrics['alignment'])
            all_metrics['entropy'].append(T_metrics['entropy'])
            all_metrics['energy'].append(T_metrics['energy'])
            all_metrics['correlation_length'].append(T_metrics['correlation_length'])
            
            # Store ensemble data for alignment
            all_metrics['alignment_ensemble'].append(np.array([T_metrics['alignment']]))
            
            # Store convergence data for this temperature
            convergence_data.append({
                'temperature': T,
                'convergence_infos': convergence_infos,
                'final_diff': convergence_infos[-1]['final_diff'] if convergence_infos else 0.0,
                'status': final_status,
                'iterations': convergence_infos[-1]['iterations'] if convergence_infos else n_sweeps_per_temperature
            })
            
            # Store vectors only at selected temperatures (if converged or plateau)
            if store_all_temperatures and i in snapshot_indices and final_status in ['converged', 'plateau']:
                try:
                    # Ensure vectors are properly shaped numpy arrays
                    snapshot_vectors = np.array(current_vectors, dtype=np.float32)
                    
                    if snapshot_dir:
                        # Save to disk
                        metadata = {
                            'concept': concept,
                            'encoder': encoder,
                            'anchor_language': anchor_language,
                            'include_anchor': include_anchor,
                            'final_status': final_status,
                            'temperature': T,
                            'k_neighbors': k  # Store k-NN parameter
                        }
                        _save_snapshot_to_disk(snapshot_dir, T, snapshot_vectors, languages, metadata)
                        print(f"[DEBUG] Saved snapshot to disk at T={T:.3f}, status={final_status}, index={i}")
                    else:
                        # Fallback to memory storage
                        vector_snapshots[T] = snapshot_vectors
                        print(f"[DEBUG] Stored snapshot in memory at T={T:.3f}, status={final_status}, index={i}")
                except Exception as e:
                    print(f"[DEBUG] Failed to save snapshot at T={T:.3f}: {e}")
            elif store_all_temperatures and i in snapshot_indices:
                print(f"[DEBUG] Skipped snapshot at T={T:.3f}, status={final_status}, index={i}")
            elif store_all_temperatures:
                print(f"[DEBUG] Index {i} not in snapshot_indices {snapshot_indices}")
            
            # Cluster analysis at this temperature
            base_threshold = sim_params.get('similarity_threshold', 0.8) if sim_params else 0.8
            temp_factor = min(1.0, max(0.5, T / 2.0))
            similarity_threshold = base_threshold + (0.15 * temp_factor)
            
            clusters = cluster_vectors(current_vectors, threshold=similarity_threshold)
            cluster_sizes = [len(cluster) for cluster in clusters]
            cluster_stats_per_temperature.append({
                'temperature': T,
                'n_clusters': len(clusters),
                'cluster_sizes': cluster_sizes,
                'clustering_threshold': similarity_threshold,
                'k_neighbors': k  # Store k-NN parameter
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
        'k_neighbors': k  # Store k-NN parameter in results
    }
    
    if store_all_temperatures:
        if snapshot_dir:
            # Store snapshot directory info for disk-based access
            result['snapshot_directory'] = snapshot_dir
            result['available_snapshot_temperatures'] = _get_available_snapshot_temperatures(snapshot_dir)
        else:
            # Fallback to memory storage
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


def create_similarity_matrix_heatmap(vectors: np.ndarray, languages: List[str] = None) -> str:
    """
    Create a diagnostic similarity matrix heatmap for debugging embedding issues.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        languages: List of language codes for labeling
        
    Returns:
        Path to saved heatmap image
    """
    try:
        # Ensure vectors are normalized
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        if languages:
            # Use language labels
            sns.heatmap(
                similarity_matrix, 
                cmap="coolwarm", 
                center=0,
                xticklabels=languages,
                yticklabels=languages,
                annot=True,
                fmt='.2f',
                square=True
            )
        else:
            # Use numeric indices
            sns.heatmap(
                similarity_matrix, 
                cmap="coolwarm", 
                center=0,
                annot=True,
                fmt='.2f',
                square=True
            )
        
        plt.title("Initial Cosine Similarity Matrix")
        plt.tight_layout()
        
        # Save to file
        output_dir = "data/diagnostics"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"similarity_matrix_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity matrix heatmap saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error creating similarity matrix heatmap: {e}")
        return None


def build_knn_graph(vectors: np.ndarray, k: int = 8) -> Dict[int, List[int]]:
    """
    Build k-nearest neighbor graph for vectors.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        k: Number of nearest neighbors to connect (default: 8)
        
    Returns:
        Dictionary mapping vector index to list of neighbor indices
    """
    # Ensure vectors are normalized
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Set diagonal to -1 to exclude self
    np.fill_diagonal(similarity_matrix, -1)
    
    # Find k nearest neighbors for each vector
    knn_graph = {}
    for i in range(len(vectors)):
        # Get indices of top k most similar vectors
        neighbors = np.argsort(similarity_matrix[i])[-k:]
        knn_graph[i] = neighbors.tolist()
    
    return knn_graph


def update_vectors_ising(
    vectors: np.ndarray, 
    T: float, 
    J: float = 1.0, 
    update_method: str = "metropolis",
    noise_sigma: float = 0.04,
    k: int = 8
) -> np.ndarray:
    """
    Apply Ising-style update logic with k-NN constraints.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength (default: 1.0)
        update_method: Update method ("metropolis" or "glauber")
        noise_sigma: Standard deviation of noise for Metropolis updates (default: 0.04)
        k: Number of nearest neighbors for k-NN constraint (default: 8)
        
    Returns:
        Updated vectors array
    """
    # Ensure vectors are normalized
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Build k-NN graph
    knn_graph = build_knn_graph(vectors, k=k)
    
    # Apply updates based on method
    if update_method == "metropolis":
        return update_vectors_metropolis_knn(vectors, T, J, noise_sigma, knn_graph)
    elif update_method == "glauber":
        return update_vectors_glauber_knn(vectors, T, J, knn_graph)
    else:
        raise ValueError(f"Unknown update method: {update_method}")


def update_vectors_metropolis_knn(
    vectors: np.ndarray, 
    T: float, 
    J: float = 1.0, 
    noise_sigma: float = 0.04,
    knn_graph: Dict[int, List[int]] = None
) -> np.ndarray:
    """
    Apply Metropolis update with k-NN constraints.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength
        noise_sigma: Standard deviation of noise
        knn_graph: k-NN graph mapping vector index to neighbor indices
        
    Returns:
        Updated vectors array
    """
    if knn_graph is None:
        knn_graph = build_knn_graph(vectors, k=8)
    
    n_vectors = len(vectors)
    updated_vectors = vectors.copy()
    
    for i in range(n_vectors):
        # Get current vector and its neighbors
        current_vector = vectors[i]
        neighbors = knn_graph[i]
        
        # Compute current energy (interaction with neighbors only)
        current_energy = 0.0
        for neighbor_idx in neighbors:
            neighbor_vector = vectors[neighbor_idx]
            # Cosine similarity (dot product for normalized vectors)
            similarity = np.dot(current_vector, neighbor_vector)
            current_energy -= J * similarity
        
        # Propose new vector with noise
        noise = np.random.normal(0, noise_sigma, current_vector.shape)
        proposed_vector = current_vector + noise
        proposed_vector = proposed_vector / np.linalg.norm(proposed_vector)
        
        # Compute proposed energy
        proposed_energy = 0.0
        for neighbor_idx in neighbors:
            neighbor_vector = vectors[neighbor_idx]
            similarity = np.dot(proposed_vector, neighbor_vector)
            proposed_energy -= J * similarity
        
        # Metropolis acceptance criterion
        energy_diff = proposed_energy - current_energy
        if energy_diff <= 0 or np.random.random() < np.exp(-energy_diff / T):
            updated_vectors[i] = proposed_vector
    
    return updated_vectors


def update_vectors_glauber_knn(
    vectors: np.ndarray, 
    T: float, 
    J: float = 1.0,
    knn_graph: Dict[int, List[int]] = None
) -> np.ndarray:
    """
    Apply Glauber update with k-NN constraints.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        T: Temperature parameter
        J: Coupling strength
        knn_graph: k-NN graph mapping vector index to neighbor indices
        
    Returns:
        Updated vectors array
    """
    if knn_graph is None:
        knn_graph = build_knn_graph(vectors, k=8)
    
    n_vectors = len(vectors)
    updated_vectors = vectors.copy()
    
    for i in range(n_vectors):
        # Get current vector and its neighbors
        current_vector = vectors[i]
        neighbors = knn_graph[i]
        
        # Compute local field from neighbors
        local_field = np.zeros_like(current_vector)
        for neighbor_idx in neighbors:
            neighbor_vector = vectors[neighbor_idx]
            local_field += J * neighbor_vector
        
        # Normalize local field
        field_norm = np.linalg.norm(local_field)
        if field_norm > 0:
            local_field = local_field / field_norm
        
        # Glauber update: move towards local field with temperature-dependent strength
        update_strength = 1.0 / (1.0 + T)  # Stronger updates at lower T
        new_vector = (1 - update_strength) * current_vector + update_strength * local_field
        new_vector = new_vector / np.linalg.norm(new_vector)
        
        updated_vectors[i] = new_vector
    
    return updated_vectors


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