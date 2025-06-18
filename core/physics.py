"""
Physics calculations for the Semantic Ising Simulator.

This module contains functions for computing physical quantities
such as system energy using consistent Hamiltonians.
"""

import numpy as np


def total_system_energy(vectors: np.ndarray, J: float = 1.0) -> float:
    """
    Compute total system energy using consistent Hamiltonian with Metropolis updates.
    
    Args:
        vectors: Array of shape (n_vectors, dim) containing normalized vectors
        J: Coupling strength parameter (default: 1.0)
    
    Returns:
        Total energy value
    
    Raises:
        ValueError: If vectors array is empty or has wrong shape
    """
    if vectors.size == 0:
        raise ValueError("Vectors array cannot be empty")
    
    if vectors.ndim != 2:
        raise ValueError("Vectors must be 2D array")
    
    n = len(vectors)
    if n < 2:
        return 0.0  # Single vector has zero energy
    
    # Use normalized vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            similarity = np.dot(normalized_vectors[i], normalized_vectors[j])
            energy -= J * similarity
    return energy 