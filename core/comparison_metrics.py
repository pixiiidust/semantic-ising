import numpy as np
from typing import Dict

# Import meta_vector computation
from .meta_vector import compute_meta_vector

def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between all pairs of vectors.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        
    Returns:
        Array of shape (n_vectors, n_vectors) containing cosine similarities
    """
    if len(vectors) == 0:
        return np.array([])
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    
    return similarity_matrix

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