import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


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
    
    # Normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarities = np.dot(normalized_vectors, normalized_vectors.T)
    
    # Create adjacency matrix (vectors are connected if similarity > threshold)
    adjacency = similarities > threshold
    
    # Find connected components (clusters)
    sparse_adj = csr_matrix(adjacency)
    n_components, labels = connected_components(sparse_adj)
    
    # Group vectors by cluster
    clusters = [[] for _ in range(n_components)]
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    # Filter clusters by minimum size
    clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
    
    return clusters 