import numpy as np
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA

"""
Meta Vector Inference Module

This module provides various methods for computing meta vectors from a set of
normalized embedding vectors. Meta vectors represent aggregate semantic
representations across multiple languages or contexts.

Functions:
    compute_centroid: Compute centroid (mean) of normalized vectors
    compute_medoid: Find medoid (closest vector to centroid)
    compute_weighted_mean: Compute weighted mean of normalized vectors
    compute_geometric_median: Compute geometric median using Weiszfeld's algorithm
    compute_first_principal_component: Compute first principal component as meta vector
    compute_meta_vector: Main meta vector computation with method selection
"""


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """
    Compute centroid (mean) of normalized vectors.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        
    Returns:
        Centroid vector of shape (dim,)
        
    Raises:
        ValueError: If vectors array is empty
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid of empty vector array")
    
    # Normalize vectors to ensure they are unit length
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute centroid (mean)
    centroid = np.mean(normalized_vectors, axis=0)
    
    # Normalize the centroid
    return centroid / np.linalg.norm(centroid)


def compute_medoid(vectors: np.ndarray) -> np.ndarray:
    """
    Find medoid (closest vector to centroid).
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        
    Returns:
        Medoid vector of shape (dim,)
        
    Raises:
        ValueError: If vectors array is empty
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute medoid of empty vector array")
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute centroid
    centroid = np.mean(normalized_vectors, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    # Find the vector closest to centroid
    distances = [np.linalg.norm(v - centroid) for v in normalized_vectors]
    medoid_idx = np.argmin(distances)
    
    return normalized_vectors[medoid_idx]


def compute_weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted mean of normalized vectors.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        weights: Array of shape (n_vectors,) containing weights
        
    Returns:
        Weighted vector of shape (dim,)
        
    Raises:
        ValueError: If vectors array is empty or weights don't match vectors length
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute weighted mean of empty vector array")
    
    if len(weights) != len(vectors):
        raise ValueError("Weights and vectors must have same length")
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Compute weighted centroid
    weighted_centroid = np.average(normalized_vectors, axis=0, weights=weights)
    
    # Normalize the result
    return weighted_centroid / np.linalg.norm(weighted_centroid)


def compute_geometric_median(vectors: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Compute geometric median using Weiszfeld's algorithm.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        max_iter: Maximum number of iterations for convergence
        
    Returns:
        Geometric median vector of shape (dim,)
        
    Raises:
        ValueError: If vectors array is empty
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute geometric median of empty vector array")
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Initialize with centroid
    median = np.mean(normalized_vectors, axis=0)
    median = median / np.linalg.norm(median)
    
    for _ in range(max_iter):
        # Compute distances
        distances = [np.linalg.norm(v - median) for v in normalized_vectors]
        
        # Avoid division by zero
        distances = [max(d, 1e-10) for d in distances]
        
        # Compute weights
        weights = [1/d for d in distances]
        weights = np.array(weights) / np.sum(weights)
        
        # Update median
        new_median = np.average(normalized_vectors, axis=0, weights=weights)
        new_median = new_median / np.linalg.norm(new_median)
        
        # Check convergence
        if np.linalg.norm(new_median - median) < 1e-6:
            break
            
        median = new_median
    
    return median


def compute_first_principal_component(vectors: np.ndarray) -> np.ndarray:
    """
    Compute first principal component as meta vector.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        
    Returns:
        First principal component vector of shape (dim,)
        
    Raises:
        ValueError: If vectors array is empty
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute principal component of empty vector array")
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Use sklearn PCA (now imported at module level)
    pca = PCA(n_components=1)
    pca.fit(normalized_vectors)
    
    # Return first principal component (already normalized)
    return pca.components_[0]


def compute_meta_vector(
    vectors: np.ndarray, 
    method: str = "centroid", 
    weights: Optional[np.ndarray] = None, 
    anchor_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute meta vector using specified method.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        method: Method to use ("centroid", "medoid", "weighted_mean", 
                "geometric_median", "first_principal_component")
        weights: Array of weights for weighted_mean method
        anchor_idx: Index of anchor vector to include in result
        
    Returns:
        Dictionary containing:
            - meta_vector: Computed meta vector
            - method: Method used
            - anchor_vector: Anchor vector if anchor_idx is valid, None otherwise
            
    Raises:
        ValueError: If method is unknown or weights required but not provided
    """
    if method == "centroid":
        meta_vector = compute_centroid(vectors)
    elif method == "medoid":
        meta_vector = compute_medoid(vectors)
    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("Weights required for weighted_mean method")
        meta_vector = compute_weighted_mean(vectors, weights)
    elif method == "geometric_median":
        meta_vector = compute_geometric_median(vectors)
    elif method == "first_principal_component":
        meta_vector = compute_first_principal_component(vectors)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get anchor vector if index is valid
    anchor_vector = None
    if anchor_idx is not None and 0 <= anchor_idx < len(vectors):
        anchor_vector = vectors[anchor_idx]
    
    result = {
        'meta_vector': meta_vector,
        'method': method,
        'anchor_vector': anchor_vector
    }
    
    return result 