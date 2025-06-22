import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform


def cluster_vectors(vectors: np.ndarray, threshold: float = 0.8, min_cluster_size: int = 2) -> list:
    """
    Cluster vectors based on cosine similarity threshold
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        threshold: Cosine similarity threshold for clustering (default: 0.8)
        min_cluster_size: Minimum size for a cluster to be included (default: 2)
        
    Returns:
        List of cluster indices, where each cluster is a list of vector indices
    """
    if len(vectors) == 0:
        return []
    
    n = len(vectors)
    print(f"DEBUG: Clustering {n} vectors with threshold {threshold}")
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarities = np.dot(normalized_vectors, normalized_vectors.T)
    
    # Debug: Print similarity statistics
    upper_tri = similarities[np.triu_indices(n, k=1)]
    print(f"DEBUG: Similarity stats - min: {upper_tri.min():.3f}, max: {upper_tri.max():.3f}, mean: {upper_tri.mean():.3f}")
    
    # Create adjacency matrix (vectors are connected if similarity > threshold)
    adjacency = similarities > threshold
    
    # Find connected components (clusters)
    sparse_adj = csr_matrix(adjacency)
    n_components, labels = connected_components(sparse_adj)
    
    print(f"DEBUG: Found {n_components} connected components")
    
    # Group vectors by cluster
    clusters = [[] for _ in range(n_components)]
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    # Filter clusters by minimum size
    original_clusters = clusters.copy()
    clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
    
    print(f"DEBUG: After filtering (min_size={min_cluster_size}): {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"DEBUG: Cluster {i+1}: {len(cluster)} vectors")
    
    return clusters


def compute_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for vectors in spin space.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        
    Returns:
        Similarity matrix of shape (n_vectors, n_vectors)
    """
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarities = np.dot(normalized_vectors, normalized_vectors.T)
    
    return similarities


def adaptive_threshold(similarities: np.ndarray, temperature: float, critical_temperature: float = None, percentile: float = 75.0) -> float:
    """
    Compute adaptive threshold based on similarity distribution using percentiles.
    
    Args:
        similarities: Similarity matrix
        temperature: Current temperature (affects threshold sensitivity)
        critical_temperature: Critical temperature (Tc) for phase boundary (optional)
        percentile: Percentile of similarity distribution to use as threshold
        
    Returns:
        Adaptive threshold value
    """
    # Get upper triangular part (excluding diagonal)
    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
    
    # Compute quartiles and percentiles for robust thresholding
    q25 = np.percentile(upper_tri, 25)  # First quartile
    q50 = np.percentile(upper_tri, 50)  # Median
    q75 = np.percentile(upper_tri, 75)  # Third quartile
    q90 = np.percentile(upper_tri, 90)  # 90th percentile
    
    print(f"DEBUG: Similarity quartiles - Q25: {q25:.3f}, Q50: {q50:.3f}, Q75: {q75:.3f}, Q90: {q90:.3f}")
    
    # PERCENTILE-BASED THRESHOLDING
    # Use different percentiles based on temperature and similarity distribution
    
    if critical_temperature is not None:
        # Use critical temperature to determine phase
        if temperature < critical_temperature:
            # ORDERED PHASE: Use higher percentile (Q75 or Q90)
            if q75 > 0.6:
                # High similarity data - use Q90 for more restrictive clustering
                base_threshold = q90
                print(f"DEBUG: Ordered phase (T={temperature:.3f} < Tc={critical_temperature:.3f}) - using Q90: {base_threshold:.3f}")
            else:
                # Moderate similarity data - use Q75
                base_threshold = q75
                print(f"DEBUG: Ordered phase (T={temperature:.3f} < Tc={critical_temperature:.3f}) - using Q75: {base_threshold:.3f}")
        else:
            # DISORDERED PHASE: Use lower percentile (Q50 or Q75)
            if q50 > 0.3:
                # Moderate disorder - use Q75
                base_threshold = q75
                print(f"DEBUG: Disordered phase (T={temperature:.3f} >= Tc={critical_temperature:.3f}) - using Q75: {base_threshold:.3f}")
            else:
                # High disorder - use Q50
                base_threshold = q50
                print(f"DEBUG: Disordered phase (T={temperature:.3f} >= Tc={critical_temperature:.3f}) - using Q50: {base_threshold:.3f}")
    else:
        # No critical temperature available - use similarity statistics
        mean_similarity = np.mean(upper_tri)
        std_similarity = np.std(upper_tri)
        
        if mean_similarity > 0.5 and std_similarity < 0.2:
            # High similarity, low variance - ordered phase
            base_threshold = q75
            print(f"DEBUG: Estimated ordered phase (mean={mean_similarity:.3f}, std={std_similarity:.3f}) - using Q75: {base_threshold:.3f}")
        elif mean_similarity < 0.2 and std_similarity > 0.3:
            # Low similarity, high variance - disordered phase
            base_threshold = q50
            print(f"DEBUG: Estimated disordered phase (mean={mean_similarity:.3f}, std={std_similarity:.3f}) - using Q50: {base_threshold:.3f}")
        else:
            # Mixed state - use Q75 as default
            base_threshold = q75
            print(f"DEBUG: Mixed state (mean={mean_similarity:.3f}, std={std_similarity:.3f}) - using Q75: {base_threshold:.3f}")
    
    # Temperature-dependent adjustment (gentle)
    temp_factor = 1.0 - 0.02 * np.log(1 + temperature)  # Very gentle temperature dependence
    adaptive_threshold = base_threshold * temp_factor
    
    # Ensure threshold is within reasonable bounds
    # Use quartiles to set bounds dynamically, but handle edge cases
    if q25 < 0:
        # If Q25 is negative, use a reasonable minimum
        min_threshold = max(0.1, q50 * 0.5)  # At least 0.1, or half of median
    else:
        min_threshold = max(0.1, q25)  # At least Q25, but minimum 0.1
    
    if q90 < 0.1:
        # If Q90 is very low, use a reasonable maximum
        max_threshold = max(0.3, q75 * 1.5)  # At least 0.3, or 1.5x Q75
    else:
        max_threshold = min(0.95, q90)  # At most Q90, but maximum 0.95
    
    # Ensure min_threshold <= max_threshold
    if min_threshold > max_threshold:
        # If bounds are inverted, use the median as a reasonable default
        min_threshold = max(0.1, q50 * 0.8)
        max_threshold = max(0.3, q50 * 1.2)
        print(f"DEBUG: Bounds were inverted, using median-based bounds")
    
    adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
    
    print(f"DEBUG: Adaptive threshold - base: {base_threshold:.3f}, temp_factor: {temp_factor:.3f}, final: {adaptive_threshold:.3f}")
    print(f"DEBUG: Threshold bounds - min: {min_threshold:.3f}, max: {max_threshold:.3f}")
    
    return adaptive_threshold


def ising_compatible_clustering(vectors: np.ndarray, temperature: float, critical_temperature: float = None, min_cluster_size: int = 2) -> Dict[str, Any]:
    """
    Perform Ising-compatible clustering in spin space (original vector space).
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing the vectors
        temperature: Current temperature for adaptive thresholding
        critical_temperature: Critical temperature (Tc) for phase boundary
        min_cluster_size: Minimum size for a cluster to be included
        
    Returns:
        Dictionary containing:
        - clusters: List of cluster indices
        - cluster_labels: Array of cluster assignments for each vector
        - cluster_entropy: Entropy of cluster size distribution
        - largest_cluster_size: Size of largest cluster (order parameter)
        - n_clusters: Number of clusters
        - threshold: Threshold used for clustering
    """
    if len(vectors) == 0:
        return {
            'clusters': [],
            'cluster_labels': np.array([]),
            'cluster_entropy': 0.0,
            'largest_cluster_size': 0,
            'n_clusters': 0,
            'threshold': 0.0
        }
    
    n_vectors = len(vectors)
    print(f"DEBUG: Ising clustering {n_vectors} vectors at T={temperature:.3f}")
    
    # Step 1: Compute similarity matrix in spin space
    similarities = compute_cosine_similarity_matrix(vectors)
    
    # Step 2: Use adaptive threshold based on temperature and data
    threshold = adaptive_threshold(similarities, temperature, critical_temperature)
    
    # Step 3: Cluster in spin space using graph theory
    # Use minimum cluster size of 2 for language data (more meaningful clusters)
    clusters = cluster_vectors(vectors, threshold=threshold, min_cluster_size=min_cluster_size)
    
    # Step 4: Create cluster labels array
    cluster_labels = np.zeros(n_vectors, dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for vector_idx in cluster:
            cluster_labels[vector_idx] = cluster_idx
    
    # Step 5: Compute cluster statistics as order parameters
    n_clusters = len(clusters)
    cluster_sizes = [len(cluster) for cluster in clusters]
    
    # Cluster entropy (distribution uniformity)
    if n_clusters > 1:
        cluster_probs = np.array(cluster_sizes) / n_vectors
        cluster_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
    else:
        cluster_entropy = 0.0
    
    # Largest cluster size (order parameter)
    largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    
    print(f"DEBUG: Ising clustering results - n_clusters: {n_clusters}, entropy: {cluster_entropy:.3f}, largest: {largest_cluster_size}")
    
    return {
        'clusters': clusters,
        'cluster_labels': cluster_labels,
        'cluster_entropy': cluster_entropy,
        'largest_cluster_size': largest_cluster_size,
        'n_clusters': n_clusters,
        'threshold': threshold
    }


def track_cluster_evolution(temperature_sweep_results: List[tuple]) -> List[Dict[str, Any]]:
    """
    Track how clusters evolve with temperature - this is the phase transition signature.
    
    Args:
        temperature_sweep_results: List of (temperature, vectors) tuples
        
    Returns:
        List of cluster evolution data for each temperature
    """
    cluster_evolution = []
    
    for temp, vectors in temperature_sweep_results:
        clustering_result = ising_compatible_clustering(vectors, temp)
        
        evolution_data = {
            'temperature': temp,
            'n_clusters': clustering_result['n_clusters'],
            'cluster_entropy': clustering_result['cluster_entropy'],
            'largest_cluster_size': clustering_result['largest_cluster_size'],
            'threshold': clustering_result['threshold'],
            'cluster_labels': clustering_result['cluster_labels']
        }
        
        cluster_evolution.append(evolution_data)
    
    return cluster_evolution


def cluster_vectors_kmeans(
    vectors: np.ndarray, 
    n_clusters: Optional[int] = None,
    random_state: int = 42
) -> tuple:
    """
    Cluster vectors using k-means algorithm.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        n_clusters: Number of clusters (auto-detect if None)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster_labels, clusters) where:
        - cluster_labels: List of cluster assignments (one per vector)
        - clusters: List of cluster assignments (list of lists)
        
    Raises:
        ValueError: If vectors array is empty or n_clusters is invalid
    """
    if len(vectors) == 0:
        raise ValueError("Cannot cluster empty vectors array")
    
    n_vectors = len(vectors)
    
    # Auto-detect number of clusters if not specified
    if n_clusters is None:
        # Use elbow method or simple heuristic
        n_clusters = min(3, n_vectors)  # Default to 3 clusters or number of vectors
    
    # Validate n_clusters
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > n_vectors:
        raise ValueError("n_clusters cannot be greater than number of vectors")
    
    # Handle edge cases
    if n_vectors == 1:
        return ([0], [[0]])
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_vectors)
    
    # Group vectors by cluster
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    
    # Remove empty clusters
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    
    return cluster_labels, clusters 