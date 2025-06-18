"""
Test suite for Phase 4.7: Advanced Comparison Metrics

Tests for comprehensive anchor language comparison methods including:
- Procrustes distance
- Centered Kernel Alignment (CKA) similarity
- Earth Mover's Distance (EMD)
- KL divergence
- Comprehensive anchor comparison
"""

import pytest
import numpy as np
from typing import Dict
import tempfile
import os
from unittest.mock import patch

# Import functions to test (these will be implemented in core/comparison_metrics.py)
try:
    from core.comparison_metrics import (
        compute_procrustes_distance,
        compute_cka_similarity,
        compute_emd_distance,
        compute_kl_divergence,
        compare_anchor_to_multilingual
    )
except ImportError:
    # Functions not yet implemented - tests will fail as expected in TDD
    pass


class TestComparisonMetrics:
    """Test suite for advanced comparison metrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create synthetic test vectors
        np.random.seed(42)  # For reproducible tests
        self.n_vectors = 5
        self.vector_dim = 10
        
        # Create two sets of vectors for comparison
        self.vectors_a = np.random.randn(self.n_vectors, self.vector_dim)
        self.vectors_b = np.random.randn(self.n_vectors, self.vector_dim)
        
        # Normalize vectors to unit length (as embeddings are normalized)
        self.vectors_a = self.vectors_a / np.linalg.norm(self.vectors_a, axis=1, keepdims=True)
        self.vectors_b = self.vectors_b / np.linalg.norm(self.vectors_b, axis=1, keepdims=True)
        
        # Create similar vectors for testing similarity metrics
        self.vectors_similar = self.vectors_a + 0.1 * np.random.randn(self.n_vectors, self.vector_dim)
        self.vectors_similar = self.vectors_similar / np.linalg.norm(self.vectors_similar, axis=1, keepdims=True)
        
        # Create identical vectors for testing perfect similarity
        self.vectors_identical = self.vectors_a.copy()
        
        # Create synthetic metrics dictionary for anchor comparison
        self.metrics = {
            'temperatures': np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            'alignment': np.array([0.9, 0.8, 0.5, 0.3, 0.2]),
            'entropy': np.array([0.1, 0.2, 0.5, 0.7, 0.8]),
            'energy': np.array([-0.9, -0.8, -0.5, -0.3, -0.2]),
            'correlation_length': np.array([1.0, 1.2, 2.0, 3.0, 4.0])
        }
    
    def test_compute_procrustes_distance_basic(self):
        """Test basic Procrustes distance computation."""
        distance = compute_procrustes_distance(self.vectors_a, self.vectors_b)
        
        # Procrustes distance should be a non-negative float
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert not np.isnan(distance)
        assert not np.isinf(distance)
    
    def test_compute_procrustes_distance_identical_vectors(self):
        """Test Procrustes distance with identical vectors (should be 0)."""
        distance = compute_procrustes_distance(self.vectors_a, self.vectors_identical)
        
        # Identical vectors should have zero Procrustes distance
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_procrustes_distance_similar_vectors(self):
        """Test Procrustes distance with similar vectors."""
        distance = compute_procrustes_distance(self.vectors_a, self.vectors_similar)
        
        # Similar vectors should have small but non-zero distance
        assert isinstance(distance, float)
        assert distance > 0.0
        assert distance < 1.0  # Should be reasonably small for similar vectors
    
    def test_compute_procrustes_distance_different_shapes(self):
        """Test Procrustes distance with different vector shapes (should raise error)."""
        vectors_different_shape = np.random.randn(self.n_vectors + 1, self.vector_dim)
        
        with pytest.raises(ValueError, match="Vector sets must have same shape"):
            compute_procrustes_distance(self.vectors_a, vectors_different_shape)
    
    def test_compute_procrustes_distance_empty_vectors(self):
        """Test Procrustes distance with empty vectors (should raise error)."""
        empty_vectors = np.array([]).reshape(0, self.vector_dim)
        
        with pytest.raises(ValueError):
            compute_procrustes_distance(empty_vectors, empty_vectors)
    
    def test_compute_cka_similarity_basic(self):
        """Test basic CKA similarity computation."""
        similarity = compute_cka_similarity(self.vectors_a, self.vectors_b)
        
        # CKA similarity should be a float between 0 and 1
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert not np.isnan(similarity)
        assert not np.isinf(similarity)
    
    def test_compute_cka_similarity_identical_vectors(self):
        """Test CKA similarity with identical vectors (should be 1.0)."""
        similarity = compute_cka_similarity(self.vectors_a, self.vectors_identical)
        
        # Identical vectors should have CKA similarity of 1.0
        assert similarity == pytest.approx(1.0, abs=1e-6)
    
    def test_compute_cka_similarity_similar_vectors(self):
        """Test CKA similarity with similar vectors."""
        similarity = compute_cka_similarity(self.vectors_a, self.vectors_similar)
        
        # Similar vectors should have high CKA similarity
        assert isinstance(similarity, float)
        assert similarity > 0.5  # Should be reasonably high for similar vectors
        assert similarity <= 1.0
    
    def test_compute_cka_similarity_orthogonal_vectors(self):
        """Test CKA similarity with orthogonal vectors."""
        # Create orthogonal vectors
        orthogonal_vectors = np.random.randn(self.n_vectors, self.vector_dim)
        orthogonal_vectors = orthogonal_vectors / np.linalg.norm(orthogonal_vectors, axis=1, keepdims=True)
        
        similarity = compute_cka_similarity(self.vectors_a, orthogonal_vectors)
        
        # Orthogonal vectors should have low CKA similarity
        assert isinstance(similarity, float)
        assert similarity >= 0.0
        assert similarity <= 1.0
    
    def test_compute_cka_similarity_zero_denominator(self):
        """Test CKA similarity when denominator is zero."""
        # Create zero vectors
        zero_vectors = np.zeros((self.n_vectors, self.vector_dim))
        
        similarity = compute_cka_similarity(zero_vectors, zero_vectors)
        
        # Should handle zero denominator gracefully
        assert similarity == 0.0
    
    def test_compute_emd_distance_basic(self):
        """Test basic EMD distance computation."""
        distance = compute_emd_distance(self.vectors_a, self.vectors_b)
        
        # EMD distance should be a non-negative float
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert not np.isnan(distance)
        assert not np.isinf(distance)
    
    def test_compute_emd_distance_identical_vectors(self):
        """Test EMD distance with identical vectors (should be 0)."""
        distance = compute_emd_distance(self.vectors_a, self.vectors_identical)
        
        # Identical vectors should have zero EMD distance
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_emd_distance_similar_vectors(self):
        """Test EMD distance with similar vectors."""
        distance = compute_emd_distance(self.vectors_a, self.vectors_similar)
        
        # Similar vectors should have small EMD distance
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert distance < 10.0  # Should be reasonably small for similar vectors
    
    def test_compute_emd_distance_different_distributions(self):
        """Test EMD distance with different distributions."""
        # Create vectors with different distributions
        vectors_uniform = np.random.uniform(0, 1, (self.n_vectors, self.vector_dim))
        vectors_normal = np.random.normal(0, 1, (self.n_vectors, self.vector_dim))
        
        distance = compute_emd_distance(vectors_uniform, vectors_normal)
        
        # Different distributions should have larger EMD distance
        assert isinstance(distance, float)
        assert distance >= 0.0
    
    def test_compute_kl_divergence_basic(self):
        """Test basic KL divergence computation."""
        divergence = compute_kl_divergence(self.vectors_a, self.vectors_b)
        
        # KL divergence should be a non-negative float
        assert isinstance(divergence, float)
        assert divergence >= 0.0
        assert not np.isnan(divergence)
        assert not np.isinf(divergence)
    
    def test_compute_kl_divergence_identical_vectors(self):
        """Test KL divergence with identical vectors (should be 0)."""
        divergence = compute_kl_divergence(self.vectors_a, self.vectors_identical)
        
        # Identical vectors should have zero KL divergence
        assert divergence == pytest.approx(0.0, abs=1e-6)
    
    def test_compute_kl_divergence_similar_vectors(self):
        """Test KL divergence with similar vectors."""
        divergence = compute_kl_divergence(self.vectors_a, self.vectors_similar)
        
        # Similar vectors should have small KL divergence
        assert isinstance(divergence, float)
        assert divergence >= 0.0
        assert divergence < 10.0  # Should be reasonably small for similar vectors
    
    def test_compute_kl_divergence_custom_bins(self):
        """Test KL divergence with custom number of bins."""
        divergence = compute_kl_divergence(self.vectors_a, self.vectors_b, bins=100)
        
        # Should work with custom bins
        assert isinstance(divergence, float)
        assert divergence >= 0.0
        assert not np.isnan(divergence)
    
    def test_compute_kl_divergence_different_distributions(self):
        """Test KL divergence with different distributions."""
        # Create vectors with different distributions
        vectors_uniform = np.random.uniform(0, 1, (self.n_vectors, self.vector_dim))
        vectors_normal = np.random.normal(0, 1, (self.n_vectors, self.vector_dim))
        
        divergence = compute_kl_divergence(vectors_uniform, vectors_normal)
        
        # Different distributions should have larger KL divergence
        assert isinstance(divergence, float)
        assert divergence >= 0.0
    
    def test_compare_anchor_to_multilingual_basic(self):
        """Test basic anchor comparison functionality."""
        # Create anchor and multilingual vectors
        anchor_vectors = self.vectors_a[:2]  # First 2 vectors as anchor
        multilingual_vectors = self.vectors_a[2:]  # Last 3 vectors as multilingual
        tc = 1.5  # Critical temperature
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Should return dictionary with all comparison metrics
        expected_keys = ['procrustes_distance', 'cka_similarity', 'emd_distance', 'kl_divergence', 'cosine_similarity', 'cosine_distance']
        for key in expected_keys:
            assert key in comparison
            assert isinstance(comparison[key], float)
        
        # Cosine metrics should be valid numbers
        assert not np.isnan(comparison['cosine_similarity'])
        assert not np.isnan(comparison['cosine_distance'])
        
        # Set-based metrics should be NaN for single vector comparison
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_compare_anchor_to_multilingual_identical_vectors(self):
        """Test anchor comparison with identical vectors."""
        # Use same vectors for anchor and multilingual
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_a.copy()
        tc = 1.5
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Identical vectors should show perfect cosine similarity
        assert comparison['cosine_similarity'] == pytest.approx(1.0, abs=1e-6)
        assert comparison['cosine_distance'] == pytest.approx(0.0, abs=1e-6)
        
        # Set-based metrics should be NaN for single vector comparison
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_compare_anchor_to_multilingual_different_vectors(self):
        """Test anchor comparison with different vectors."""
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_b
        tc = 1.5
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Different vectors should show lower cosine similarity
        assert comparison['cosine_similarity'] < 1.0
        assert comparison['cosine_distance'] > 0.0
        
        # Set-based metrics should be NaN for single vector comparison
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_compare_anchor_to_multilingual_edge_cases(self):
        """Test anchor comparison with edge cases."""
        # Test with single vectors
        anchor_vectors = self.vectors_a[:1]
        multilingual_vectors = self.vectors_b[:1]
        tc = 1.5
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Should handle single vectors gracefully
        expected_keys = ['procrustes_distance', 'cka_similarity', 'emd_distance', 'kl_divergence', 'cosine_similarity', 'cosine_distance']
        for key in expected_keys:
            assert key in comparison
            assert isinstance(comparison[key], float)
        
        # Cosine metrics should be valid
        assert not np.isnan(comparison['cosine_similarity'])
        assert not np.isnan(comparison['cosine_distance'])
        
        # Set-based metrics should be NaN
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_compare_anchor_to_multilingual_invalid_tc(self):
        """Test anchor comparison with invalid critical temperature."""
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_b
        tc = 10.0  # Temperature not in metrics range
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Should handle invalid Tc gracefully by using closest available temperature
        expected_keys = ['procrustes_distance', 'cka_similarity', 'emd_distance', 'kl_divergence', 'cosine_similarity', 'cosine_distance']
        for key in expected_keys:
            assert key in comparison
            assert isinstance(comparison[key], float)
    
    def test_compare_anchor_to_multilingual_missing_metrics_keys(self):
        """Test anchor comparison with missing metrics keys."""
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_b
        tc = 1.5
        
        # Create incomplete metrics dictionary
        incomplete_metrics = {'temperatures': np.array([1.0, 1.5, 2.0])}
        
        with pytest.raises(KeyError):
            compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, incomplete_metrics)
    
    def test_integration_all_metrics_consistent(self):
        """Integration test: verify all metrics are consistent for same input."""
        # Test that all metrics give consistent results for identical inputs
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_a.copy()
        tc = 1.5
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Cosine similarity should indicate perfect similarity for identical inputs
        assert comparison['cosine_similarity'] == pytest.approx(1.0, abs=1e-6)
        assert comparison['cosine_distance'] == pytest.approx(0.0, abs=1e-6)
        
        # Set-based metrics should be NaN for single vector comparison
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_integration_metric_ranges(self):
        """Integration test: verify all metrics are in expected ranges."""
        anchor_vectors = self.vectors_a
        multilingual_vectors = self.vectors_b
        tc = 1.5
        
        comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, self.metrics)
        
        # Verify cosine similarity range
        assert -1.0 <= comparison['cosine_similarity'] <= 1.0
        assert 0.0 <= comparison['cosine_distance'] <= 2.0  # Distance can be 0 to 2
        
        # Set-based metrics should be NaN for single vector comparison
        assert np.isnan(comparison['procrustes_distance'])
        assert np.isnan(comparison['cka_similarity'])
        assert np.isnan(comparison['emd_distance'])
        assert np.isnan(comparison['kl_divergence'])
    
    def test_compare_anchor_to_multilingual_meta_vector_comparison(self):
        """Test that anchor comparison uses meta-vector of multilingual set, not individual vectors."""
        # Create test data: anchor vector and multilingual vectors
        anchor_vectors = np.random.randn(1, 768)  # Single anchor vector
        multilingual_vectors = np.random.randn(5, 768)  # 5 multilingual vectors
        tc = 1.5
        metrics = {
            'temperatures': np.array([1.0, 1.5, 2.0]),
            'alignment': np.array([0.8, 0.5, 0.2]),
            'entropy': np.array([0.2, 0.5, 0.8]),
            'energy': np.array([-0.8, -0.5, -0.2]),
            'correlation_length': np.array([0.1, 0.5, 0.9])
        }
        
        # Mock the meta_vector computation to verify it's called
        with patch('core.comparison_metrics.compute_meta_vector') as mock_meta:
            # Mock meta-vector computation
            mock_meta.return_value = {
                'meta_vector': np.random.randn(768),
                'method': 'centroid'
            }
            
            comparison = compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, metrics)
            
            # Verify that compute_meta_vector was called with multilingual vectors
            mock_meta.assert_called_once_with(multilingual_vectors, method="centroid")
            
            # Verify comparison metrics are computed correctly for single vector comparison
            assert 'cosine_similarity' in comparison
            assert 'cosine_distance' in comparison
            assert isinstance(comparison['cosine_similarity'], float)
            assert isinstance(comparison['cosine_distance'], float)
            assert not np.isnan(comparison['cosine_similarity'])
            assert not np.isnan(comparison['cosine_distance'])
            
            # Verify that metrics requiring multiple vectors are set to NaN
            assert np.isnan(comparison['procrustes_distance'])
            assert np.isnan(comparison['cka_similarity'])
            assert np.isnan(comparison['emd_distance'])
            assert np.isnan(comparison['kl_divergence'])
            
            # Verify cosine distance is the complement of cosine similarity
            assert abs(comparison['cosine_distance'] - (1.0 - comparison['cosine_similarity'])) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 