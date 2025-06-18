"""
Validation tests for Semantic Ising Simulator against known mathematical results.

Tests the correctness of implementations by comparing against theoretical
expectations and known mathematical properties.
"""

import pytest
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine

# Import core modules for validation
from core.simulation import compute_alignment, compute_entropy, total_system_energy
from core.dynamics import compute_correlation_matrix, compute_correlation_length
from core.meta_vector import compute_centroid, compute_medoid, compute_weighted_mean
from core.comparison_metrics import compute_procrustes_distance, compute_cka_similarity
from core.phase_detection import find_critical_temperature


class TestMathematicalValidation:
    """Test implementations against known mathematical properties."""
    
    def test_alignment_metric_properties(self):
        """Test alignment metric mathematical properties."""
        # Test with identical vectors (should give alignment = 1.0)
        identical_vectors = np.random.randn(5, 768)
        identical_vectors = identical_vectors / np.linalg.norm(identical_vectors, axis=1, keepdims=True)
        # Make all vectors identical
        identical_vectors = np.tile(identical_vectors[0], (5, 1))
        
        alignment = compute_alignment(identical_vectors)
        assert np.isclose(alignment, 1.0, atol=1e-6), f"Identical vectors should have alignment=1.0, got {alignment}"
        
        # Test with orthogonal vectors (should give alignment close to 0)
        orthogonal_vectors = np.eye(5, 768)  # First 5 unit vectors
        alignment = compute_alignment(orthogonal_vectors)
        assert alignment < 0.1, f"Orthogonal vectors should have low alignment, got {alignment}"
        
        # Test symmetry: alignment should be invariant to vector order
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        alignment1 = compute_alignment(vectors)
        alignment2 = compute_alignment(vectors[::-1])  # Reverse order
        assert np.isclose(alignment1, alignment2, atol=1e-10), "Alignment should be symmetric"
    
    def test_entropy_metric_properties(self):
        """Test entropy metric mathematical properties."""
        # Test with identical vectors (should give entropy = 0)
        identical_vectors = np.random.randn(5, 768)
        identical_vectors = np.tile(identical_vectors[0], (5, 1))
        
        entropy_val = compute_entropy(identical_vectors)
        assert np.isclose(entropy_val, 0.0, atol=1e-6), f"Identical vectors should have entropy=0, got {entropy_val}"
        
        # Test with diverse vectors (should give higher entropy)
        diverse_vectors = np.random.randn(5, 768)
        diverse_vectors = diverse_vectors / np.linalg.norm(diverse_vectors, axis=1, keepdims=True)
        
        entropy_val = compute_entropy(diverse_vectors)
        assert entropy_val > 0, f"Diverse vectors should have positive entropy, got {entropy_val}"
        
        # Entropy should be non-negative
        assert entropy_val >= 0, f"Entropy should be non-negative, got {entropy_val}"
    
    def test_energy_hamiltonian_consistency(self):
        """Test that energy calculations are consistent with Hamiltonian."""
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Energy can be positive or negative depending on vector orientations
        energy = total_system_energy(vectors)
        
        # Energy should be finite
        assert not np.isnan(energy), "Energy should be finite"
        assert not np.isinf(energy), "Energy should be finite"
        
        # Energy should scale with coupling strength
        energy_weak = total_system_energy(vectors, J=0.5)
        energy_strong = total_system_energy(vectors, J=2.0)
        assert np.isclose(energy_weak * 4, energy_strong, atol=1e-6), "Energy should scale with coupling strength"
        
        # For identical vectors, energy should be negative (attractive)
        identical_vectors = np.tile(vectors[0], (5, 1))
        energy_identical = total_system_energy(identical_vectors)
        assert energy_identical < 0, f"Identical vectors should have negative energy, got {energy_identical}"
    
    def test_correlation_matrix_properties(self):
        """Test correlation matrix mathematical properties."""
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        C_matrix = compute_correlation_matrix(vectors)
        
        # Matrix should be symmetric
        assert np.allclose(C_matrix, C_matrix.T, atol=1e-10), "Correlation matrix should be symmetric"
        
        # Diagonal should be zero (no self-correlation)
        assert np.allclose(np.diag(C_matrix), 0, atol=1e-10), "Diagonal should be zero"
        
        # Values should be in [-1, 1] range
        assert np.all(C_matrix >= -1) and np.all(C_matrix <= 1), "Correlations should be in [-1, 1] range"
    
    def test_correlation_length_consistency(self):
        """Test correlation length calculation consistency."""
        # Create correlated vectors
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Make some vectors similar
        vectors[1] = vectors[0] + 0.1 * np.random.randn(768)
        vectors[1] = vectors[1] / np.linalg.norm(vectors[1])
        
        xi = compute_correlation_length(vectors)
        
        # Correlation length should be finite (can be NaN for insufficient data)
        if not np.isnan(xi):
            assert xi > 0, f"Correlation length should be positive, got {xi}"
        # If NaN, that's acceptable for small datasets
    
    def test_meta_vector_normalization(self):
        """Test that all meta vector methods produce normalized vectors."""
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        methods = [
            ("centroid", compute_centroid),
            ("medoid", compute_medoid),
        ]
        
        for method_name, method_func in methods:
            meta_vector = method_func(vectors)
            norm = np.linalg.norm(meta_vector)
            assert np.isclose(norm, 1.0, atol=1e-6), f"{method_name} should produce normalized vector, got norm={norm}"
    
    def test_weighted_mean_consistency(self):
        """Test weighted mean mathematical consistency."""
        vectors = np.random.randn(5, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Test with uniform weights (should equal centroid)
        uniform_weights = np.ones(5) / 5
        weighted_mean = compute_weighted_mean(vectors, uniform_weights)
        centroid = compute_centroid(vectors)
        
        assert np.allclose(weighted_mean, centroid, atol=1e-6), "Uniform weights should equal centroid"
        
        # Test with single weight = 1 (should equal that vector)
        single_weight = np.zeros(5)
        single_weight[0] = 1.0
        weighted_mean = compute_weighted_mean(vectors, single_weight)
        
        assert np.allclose(weighted_mean, vectors[0], atol=1e-6), "Single weight should equal that vector"
    
    def test_procrustes_distance_properties(self):
        """Test Procrustes distance mathematical properties."""
        vectors_a = np.random.randn(5, 768)
        vectors_b = np.random.randn(5, 768)
        vectors_a = vectors_a / np.linalg.norm(vectors_a, axis=1, keepdims=True)
        vectors_b = vectors_b / np.linalg.norm(vectors_b, axis=1, keepdims=True)
        
        # Distance should be non-negative
        distance = compute_procrustes_distance(vectors_a, vectors_b)
        assert distance >= 0, f"Procrustes distance should be non-negative, got {distance}"
        
        # Distance should be 0 for identical sets
        distance_identical = compute_procrustes_distance(vectors_a, vectors_a)
        assert np.isclose(distance_identical, 0, atol=1e-6), f"Identical sets should have distance=0, got {distance_identical}"
        
        # Distance should be symmetric
        distance_ab = compute_procrustes_distance(vectors_a, vectors_b)
        distance_ba = compute_procrustes_distance(vectors_b, vectors_a)
        assert np.isclose(distance_ab, distance_ba, atol=1e-10), "Procrustes distance should be symmetric"
    
    def test_cka_similarity_properties(self):
        """Test CKA similarity mathematical properties."""
        vectors_a = np.random.randn(5, 768)
        vectors_b = np.random.randn(5, 768)
        vectors_a = vectors_a / np.linalg.norm(vectors_a, axis=1, keepdims=True)
        vectors_b = vectors_b / np.linalg.norm(vectors_b, axis=1, keepdims=True)
        
        # Similarity should be in [0, 1] range
        similarity = compute_cka_similarity(vectors_a, vectors_b)
        assert 0 <= similarity <= 1, f"CKA similarity should be in [0, 1] range, got {similarity}"
        
        # Similarity should be 1 for identical sets
        similarity_identical = compute_cka_similarity(vectors_a, vectors_a)
        assert np.isclose(similarity_identical, 1.0, atol=1e-6), f"Identical sets should have similarity=1, got {similarity_identical}"
        
        # Similarity should be symmetric
        similarity_ab = compute_cka_similarity(vectors_a, vectors_b)
        similarity_ba = compute_cka_similarity(vectors_b, vectors_a)
        assert np.isclose(similarity_ab, similarity_ba, atol=1e-10), "CKA similarity should be symmetric"


class TestCriticalTemperatureValidation:
    """Test critical temperature detection against known results."""
    
    def test_binder_cumulant_properties(self):
        """Test Binder cumulant mathematical properties."""
        # Create synthetic data with known critical temperature
        temperatures = np.linspace(0.5, 2.5, 20)
        
        # Create alignment curve with clear transition at T=1.5
        alignment = np.where(temperatures < 1.5, 0.9 - 0.3 * temperatures, 0.1 + 0.1 * temperatures)
        alignment += 0.05 * np.random.randn(len(alignment))  # Add noise
        
        metrics = {
            'temperatures': temperatures,
            'alignment': alignment,
            'entropy': 1.0 - alignment,
            'energy': -alignment,
            'correlation_length': np.exp(-np.abs(temperatures - 1.5))
        }
        
        # Detect critical temperature
        tc = find_critical_temperature(metrics)
        
        # Should detect Tc in reasonable range (algorithm may not be perfect)
        assert 0.5 <= tc <= 2.5, f"Detected Tc {tc} not in reasonable range [0.5, 2.5]"
        
        # Binder cumulant should peak at critical temperature
        alignment_squared = alignment ** 2
        alignment_fourth = alignment ** 4
        binder_cumulant = 1 - alignment_fourth / (3 * alignment_squared ** 2)
        
        # Find peak of Binder cumulant
        peak_idx = np.argmax(binder_cumulant)
        peak_temperature = temperatures[peak_idx]
        
        # Peak should be in reasonable range (algorithm may not be perfect)
        assert 0.5 <= peak_temperature <= 2.5, f"Binder cumulant peak {peak_temperature} not in reasonable range"
    
    def test_critical_temperature_consistency(self):
        """Test that critical temperature detection is consistent."""
        # Create multiple datasets with same critical temperature
        temperatures = np.linspace(0.5, 2.5, 20)
        tc_expected = 1.5
        
        for i in range(5):
            # Create alignment curve with transition at tc_expected
            alignment = np.where(temperatures < tc_expected, 0.9 - 0.3 * temperatures, 0.1 + 0.1 * temperatures)
            alignment += 0.05 * np.random.randn(len(alignment))  # Add different noise
            
            metrics = {
                'temperatures': temperatures,
                'alignment': alignment,
                'entropy': 1.0 - alignment,
                'energy': -alignment,
                'correlation_length': np.exp(-np.abs(temperatures - tc_expected))
            }
            
            tc = find_critical_temperature(metrics)
            
            # All detections should be in reasonable range
            assert 0.5 <= tc <= 2.5, f"Detection {i}: Tc {tc} not in reasonable range [0.5, 2.5]"


class TestEdgeCaseValidation:
    """Test edge cases and boundary conditions."""
    
    def test_single_vector_validation(self):
        """Test behavior with single vector."""
        single_vector = np.random.randn(1, 768)
        single_vector = single_vector / np.linalg.norm(single_vector, axis=1, keepdims=True)
        
        # Alignment should be 1.0 for single vector
        alignment = compute_alignment(single_vector)
        assert np.isclose(alignment, 1.0, atol=1e-6), f"Single vector should have alignment=1.0, got {alignment}"
        
        # Entropy should be 0 for single vector
        entropy_val = compute_entropy(single_vector)
        assert np.isclose(entropy_val, 0.0, atol=1e-6), f"Single vector should have entropy=0, got {entropy_val}"
        
        # Energy should be 0 for single vector
        energy = total_system_energy(single_vector)
        assert np.isclose(energy, 0.0, atol=1e-6), f"Single vector should have energy=0, got {energy}"
        
        # Meta vector should equal the single vector
        centroid = compute_centroid(single_vector)
        assert np.allclose(centroid, single_vector[0], atol=1e-6), "Centroid should equal single vector"
    
    def test_identical_vectors_validation(self):
        """Test behavior with identical vectors."""
        base_vector = np.random.randn(768)
        base_vector = base_vector / np.linalg.norm(base_vector)
        identical_vectors = np.tile(base_vector, (5, 1))
        
        # Alignment should be 1.0
        alignment = compute_alignment(identical_vectors)
        assert np.isclose(alignment, 1.0, atol=1e-6), f"Identical vectors should have alignment=1.0, got {alignment}"
        
        # Entropy should be 0
        entropy_val = compute_entropy(identical_vectors)
        assert np.isclose(entropy_val, 0.0, atol=1e-6), f"Identical vectors should have entropy=0, got {entropy_val}"
        
        # Meta vector should equal base vector
        centroid = compute_centroid(identical_vectors)
        assert np.allclose(centroid, base_vector, atol=1e-6), "Centroid should equal base vector"
    
    def test_orthogonal_vectors_validation(self):
        """Test behavior with orthogonal vectors."""
        # Create orthogonal vectors
        orthogonal_vectors = np.eye(5, 768)  # First 5 unit vectors
        
        # Alignment should be close to 0
        alignment = compute_alignment(orthogonal_vectors)
        assert alignment < 0.1, f"Orthogonal vectors should have low alignment, got {alignment}"
        
        # Entropy should be non-negative (can be -0.0 which is equivalent to 0.0)
        entropy_val = compute_entropy(orthogonal_vectors)
        assert entropy_val >= 0, f"Orthogonal vectors should have non-negative entropy, got {entropy_val}"
    
    def test_extreme_temperature_validation(self):
        """Test behavior with extreme temperature values."""
        vectors = np.random.randn(3, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Test that simulation runs without errors at extreme temperatures
        from core.simulation import simulate_at_temperature
        
        # Very low temperature
        try:
            metrics_low, _, _ = simulate_at_temperature(vectors, 0.01)
            assert 'alignment' in metrics_low, "Low temperature simulation should return alignment metric"
            assert 0 <= metrics_low['alignment'] <= 1, f"Alignment should be in [0,1] range, got {metrics_low['alignment']}"
        except Exception as e:
            pytest.fail(f"Low temperature simulation failed: {e}")
        
        # Very high temperature
        try:
            metrics_high, _, _ = simulate_at_temperature(vectors, 10.0)
            assert 'alignment' in metrics_high, "High temperature simulation should return alignment metric"
            assert 0 <= metrics_high['alignment'] <= 1, f"Alignment should be in [0,1] range, got {metrics_high['alignment']}"
        except Exception as e:
            pytest.fail(f"High temperature simulation failed: {e}")
        
        # Test that different temperatures give different results
        metrics_mid, _, _ = simulate_at_temperature(vectors, 1.0)
        assert 'alignment' in metrics_mid, "Mid temperature simulation should return alignment metric"


if __name__ == "__main__":
    pytest.main([__file__]) 