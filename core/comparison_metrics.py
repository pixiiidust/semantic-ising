import numpy as np
from typing import Dict
from scipy.spatial import procrustes
from scipy.stats import wasserstein_distance, entropy

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
    """Compare anchor language to multilingual result at critical temperature."""
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
    
    # Handle different shapes by using the smaller set for comparison
    min_vectors = min(len(anchor_vectors), len(tc_vectors))
    anchor_subset = anchor_vectors[:min_vectors]
    tc_subset = tc_vectors[:min_vectors]
    
    # Handle single vector case (edge case)
    if min_vectors == 1:
        # For single vectors, use cosine similarity and set other metrics to reasonable defaults
        cosine_sim = np.dot(anchor_subset[0], tc_subset[0]) / (np.linalg.norm(anchor_subset[0]) * np.linalg.norm(tc_subset[0]))
        return {
            'procrustes_distance': 1.0 - abs(cosine_sim),  # Approximate for single vectors
            'cka_similarity': abs(cosine_sim),  # Approximate for single vectors
            'emd_distance': compute_emd_distance(anchor_subset, tc_subset),
            'kl_divergence': compute_kl_divergence(anchor_subset, tc_subset),
            'cosine_similarity': cosine_sim
        }
    
    # Compute all comparison metrics for multiple vectors
    comparison = {
        'procrustes_distance': compute_procrustes_distance(anchor_subset, tc_subset),
        'cka_similarity': compute_cka_similarity(anchor_subset, tc_subset),
        'emd_distance': compute_emd_distance(anchor_subset, tc_subset),
        'kl_divergence': compute_kl_divergence(anchor_subset, tc_subset),
        'cosine_similarity': np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 
                                    for a, b in zip(anchor_subset, tc_subset)])
    }
    
    return comparison 