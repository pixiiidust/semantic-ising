import numpy as np
from typing import Dict
from scipy.spatial import procrustes
from scipy.stats import wasserstein_distance, entropy

# Import meta_vector computation
from .meta_vector import compute_meta_vector

def compute_procrustes_distance(vectors_a: np.ndarray, vectors_b: np.ndarray) -> float:
    """Compute Procrustes distance between two sets of vectors."""
    if vectors_a.shape != vectors_b.shape:
        raise ValueError("Vector sets must have same shape")
    
    if vectors_a.size == 0:
        raise ValueError("Empty vectors not supported")
    
    # Procrustes analysis returns (transformed_a, transformed_b, disparity)
    _, _, disparity = procrustes(vectors_a, vectors_b)
    return disparity

def compute_cka_similarity(vectors_a: np.ndarray, vectors_b: np.ndarray) -> float:
    """Compute Centered Kernel Alignment similarity between two sets of vectors."""
    # Center the vectors
    vectors_a_centered = vectors_a - np.mean(vectors_a, axis=0)
    vectors_b_centered = vectors_b - np.mean(vectors_b, axis=0)
    
    # Compute kernel matrices
    K_a = vectors_a_centered @ vectors_a_centered.T
    K_b = vectors_b_centered @ vectors_b_centered.T
    
    # Compute CKA
    numerator = np.trace(K_a @ K_b)
    denominator = np.sqrt(np.trace(K_a @ K_a) * np.trace(K_b @ K_b))
    
    return numerator / denominator if denominator > 0 else 0.0

def compute_emd_distance(vectors_a: np.ndarray, vectors_b: np.ndarray) -> float:
    """Compute Earth Mover's Distance between vector distributions."""
    # Flatten vectors for 1D EMD computation
    flat_a = vectors_a.flatten()
    flat_b = vectors_b.flatten()
    
    return wasserstein_distance(flat_a, flat_b)

def compute_kl_divergence(vectors_a: np.ndarray, vectors_b: np.ndarray, bins: int = 50) -> float:
    """Compute KL divergence between vector distributions."""
    # Compute histograms of flattened vectors
    flat_a = vectors_a.flatten()
    flat_b = vectors_b.flatten()
    
    hist_a, _ = np.histogram(flat_a, bins=bins, density=True)
    hist_b, _ = np.histogram(flat_b, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_a = hist_a + epsilon
    hist_b = hist_b + epsilon
    
    return entropy(hist_a, hist_b)

def compare_anchor_to_multilingual(anchor_vectors: np.ndarray, multilingual_vectors: np.ndarray, tc: float, metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compare anchor language to multilingual result at critical temperature.
    
    For single vector vs meta-vector comparison, only cosine distance and cosine similarity 
    are meaningful. Set-based metrics (Procrustes, CKA, EMD, KL) are set to NaN since 
    they require multiple vectors for meaningful computation.
    
    Args:
        anchor_vectors: Anchor language vector(s)
        multilingual_vectors: Multilingual vectors at Tc
        tc: Critical temperature
        metrics: Simulation metrics dictionary
        
    Returns:
        Dictionary with comparison metrics:
        - cosine_distance: Primary semantic distance metric (0-1, lower is better)
        - cosine_similarity: Directional similarity (0-1, higher is better)
        - procrustes_distance: NaN (requires multiple vectors)
        - cka_similarity: NaN (requires multiple vectors)
        - emd_distance: NaN (requires multiple vectors)
        - kl_divergence: NaN (requires multiple vectors)
    """
    # Validate required metrics keys
    required_keys = ['temperatures', 'alignment', 'entropy', 'energy', 'correlation_length']
    for key in required_keys:
        if key not in metrics:
            raise KeyError(f"Missing required metric: {key}")
    
    # Find index closest to Tc
    tc_idx = np.argmin(np.abs(metrics['temperatures'] - tc))
    
    # Extract vectors at Tc (if available)
    if 'vectors_at_tc' in metrics:
        tc_vectors = metrics['vectors_at_tc'][tc_idx]
    else:
        tc_vectors = multilingual_vectors
    
    # Compute meta-vector from multilingual set at Tc
    meta_result = compute_meta_vector(tc_vectors, method="centroid")
    multilingual_meta_vector = meta_result['meta_vector']
    
    # Handle multiple anchor vectors by computing their meta-vector too
    if len(anchor_vectors) > 1:
        anchor_meta_result = compute_meta_vector(anchor_vectors, method="centroid")
        anchor_vector = anchor_meta_result['meta_vector']
    else:
        # Single anchor vector
        anchor_vector = anchor_vectors[0]
    
    # Normalize vectors for cosine distance computation
    anchor_norm = np.linalg.norm(anchor_vector)
    meta_norm = np.linalg.norm(multilingual_meta_vector)
    
    if anchor_norm == 0 or meta_norm == 0:
        # Handle zero vectors
        cosine_similarity = 0.0
        cosine_distance = 1.0
    else:
        # Compute cosine similarity and distance
        cosine_similarity = np.dot(anchor_vector, multilingual_meta_vector) / (anchor_norm * meta_norm)
        cosine_distance = 1.0 - cosine_similarity
    
    # For single vector comparison, use cosine distance as primary metric
    # Set other metrics to NaN since they're designed for vector sets
    comparison = {
        'procrustes_distance': np.nan,  # Requires multiple points
        'cka_similarity': np.nan,       # Requires multiple points  
        'emd_distance': np.nan,         # Requires multiple points
        'kl_divergence': np.nan,        # Requires multiple points
        'cosine_similarity': cosine_similarity,
        'cosine_distance': cosine_distance  # Primary semantic distance metric
    }
    
    return comparison 