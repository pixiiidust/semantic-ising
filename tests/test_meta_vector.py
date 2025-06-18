import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.meta_vector import (
    compute_centroid,
    compute_medoid,
    compute_weighted_mean,
    compute_geometric_median,
    compute_first_principal_component,
    compute_meta_vector
)


class TestMetaVector:
    """Test suite for meta vector computation functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create test vectors (normalized)
        np.random.seed(42)
        self.test_vectors = np.random.randn(5, 768)
        self.test_vectors = self.test_vectors / np.linalg.norm(self.test_vectors, axis=1, keepdims=True)
        
        # Create test weights
        self.test_weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        
        # Create identical vectors for testing
        self.identical_vectors = np.ones((3, 768))
        self.identical_vectors = self.identical_vectors / np.linalg.norm(self.identical_vectors, axis=1, keepdims=True)
    
    def test_compute_centroid_basic(self):
        """Test basic centroid computation"""
        centroid = compute_centroid(self.test_vectors)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (768,)
        assert np.linalg.norm(centroid) > 0
        assert np.allclose(np.linalg.norm(centroid), 1.0, atol=1e-6)
    
    def test_compute_centroid_identical_vectors(self):
        """Test centroid with identical vectors"""
        centroid = compute_centroid(self.identical_vectors)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (768,)
        assert np.allclose(np.linalg.norm(centroid), 1.0, atol=1e-6)
        # Should be very close to the original vectors since they're identical
        assert np.allclose(centroid, self.identical_vectors[0], atol=1e-6)
    
    def test_compute_centroid_single_vector(self):
        """Test centroid with single vector"""
        single_vector = self.test_vectors[:1]
        centroid = compute_centroid(single_vector)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (768,)
        assert np.allclose(centroid, single_vector[0])
    
    def test_compute_centroid_empty_input(self):
        """Test centroid with empty input"""
        with pytest.raises(ValueError):
            compute_centroid(np.array([]))
    
    def test_compute_medoid_basic(self):
        """Test basic medoid computation"""
        medoid = compute_medoid(self.test_vectors)
        
        assert isinstance(medoid, np.ndarray)
        assert medoid.shape == (768,)
        assert np.linalg.norm(medoid) > 0
        # Medoid should be one of the original vectors
        assert any(np.allclose(medoid, v, atol=1e-6) for v in self.test_vectors)
    
    def test_compute_medoid_identical_vectors(self):
        """Test medoid with identical vectors"""
        medoid = compute_medoid(self.identical_vectors)
        
        assert isinstance(medoid, np.ndarray)
        assert medoid.shape == (768,)
        assert np.allclose(medoid, self.identical_vectors[0])
    
    def test_compute_medoid_single_vector(self):
        """Test medoid with single vector"""
        single_vector = self.test_vectors[:1]
        medoid = compute_medoid(single_vector)
        
        assert isinstance(medoid, np.ndarray)
        assert medoid.shape == (768,)
        assert np.allclose(medoid, single_vector[0])
    
    def test_compute_weighted_mean_basic(self):
        """Test basic weighted mean computation"""
        weighted_mean = compute_weighted_mean(self.test_vectors, self.test_weights)
        
        assert isinstance(weighted_mean, np.ndarray)
        assert weighted_mean.shape == (768,)
        assert np.linalg.norm(weighted_mean) > 0
        assert np.allclose(np.linalg.norm(weighted_mean), 1.0, atol=1e-6)
    
    def test_compute_weighted_mean_uniform_weights(self):
        """Test weighted mean with uniform weights"""
        uniform_weights = np.ones(len(self.test_vectors)) / len(self.test_vectors)
        weighted_mean = compute_weighted_mean(self.test_vectors, uniform_weights)
        centroid = compute_centroid(self.test_vectors)
        
        # Should be very close to centroid with uniform weights
        assert np.allclose(weighted_mean, centroid, atol=1e-6)
    
    def test_compute_weighted_mean_mismatched_weights(self):
        """Test weighted mean with mismatched weights"""
        wrong_weights = np.array([0.5, 0.5])  # Only 2 weights for 5 vectors
        with pytest.raises(ValueError, match="Weights and vectors must have same length"):
            compute_weighted_mean(self.test_vectors, wrong_weights)
    
    def test_compute_weighted_mean_negative_weights(self):
        """Test weighted mean with negative weights (should be normalized)"""
        negative_weights = np.array([-1, -2, -3, -4, -5])
        weighted_mean = compute_weighted_mean(self.test_vectors, negative_weights)
        
        assert isinstance(weighted_mean, np.ndarray)
        assert weighted_mean.shape == (768,)
        assert np.linalg.norm(weighted_mean) > 0
    
    def test_compute_geometric_median_basic(self):
        """Test basic geometric median computation"""
        geometric_median = compute_geometric_median(self.test_vectors)
        
        assert isinstance(geometric_median, np.ndarray)
        assert geometric_median.shape == (768,)
        assert np.linalg.norm(geometric_median) > 0
        assert np.allclose(np.linalg.norm(geometric_median), 1.0, atol=1e-6)
    
    def test_compute_geometric_median_identical_vectors(self):
        """Test geometric median with identical vectors"""
        geometric_median = compute_geometric_median(self.identical_vectors)
        
        assert isinstance(geometric_median, np.ndarray)
        assert geometric_median.shape == (768,)
        assert np.allclose(geometric_median, self.identical_vectors[0])
    
    def test_compute_geometric_median_convergence(self):
        """Test geometric median convergence with few iterations"""
        geometric_median = compute_geometric_median(self.test_vectors, max_iter=10)
        
        assert isinstance(geometric_median, np.ndarray)
        assert geometric_median.shape == (768,)
        assert np.linalg.norm(geometric_median) > 0
    
    def test_compute_geometric_median_single_vector(self):
        """Test geometric median with single vector"""
        single_vector = self.test_vectors[:1]
        geometric_median = compute_geometric_median(single_vector)
        
        assert isinstance(geometric_median, np.ndarray)
        assert geometric_median.shape == (768,)
        assert np.allclose(geometric_median, single_vector[0])
    
    @patch('core.meta_vector.PCA')
    def test_compute_first_principal_component_basic(self, mock_pca):
        """Test basic first principal component computation"""
        # Mock PCA
        mock_pca_instance = MagicMock()
        mock_pca_instance.components_ = np.random.randn(1, 768)
        mock_pca_instance.components_[0] = mock_pca_instance.components_[0] / np.linalg.norm(mock_pca_instance.components_[0])
        mock_pca.return_value = mock_pca_instance
        
        pca_vector = compute_first_principal_component(self.test_vectors)
        
        assert isinstance(pca_vector, np.ndarray)
        assert pca_vector.shape == (768,)
        assert np.linalg.norm(pca_vector) > 0
        mock_pca.assert_called_once_with(n_components=1)
        mock_pca_instance.fit.assert_called_once()
    
    @patch('core.meta_vector.PCA')
    def test_compute_first_principal_component_identical_vectors(self, mock_pca):
        """Test first principal component with identical vectors"""
        # Mock PCA
        mock_pca_instance = MagicMock()
        mock_pca_instance.components_ = np.random.randn(1, 768)
        mock_pca_instance.components_[0] = mock_pca_instance.components_[0] / np.linalg.norm(mock_pca_instance.components_[0])
        mock_pca.return_value = mock_pca_instance
        
        pca_vector = compute_first_principal_component(self.identical_vectors)
        
        assert isinstance(pca_vector, np.ndarray)
        assert pca_vector.shape == (768,)
        assert np.linalg.norm(pca_vector) > 0
    
    def test_compute_first_principal_component_single_vector(self):
        """Test first principal component with single vector"""
        single_vector = self.test_vectors[:1]
        pca_vector = compute_first_principal_component(single_vector)
        
        assert isinstance(pca_vector, np.ndarray)
        assert pca_vector.shape == (768,)
        assert np.linalg.norm(pca_vector) > 0
    
    def test_compute_meta_vector_centroid(self):
        """Test meta vector computation with centroid method"""
        result = compute_meta_vector(self.test_vectors, method="centroid")
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "centroid"
        assert isinstance(result['meta_vector'], np.ndarray)
        assert result['meta_vector'].shape == (768,)
        assert result['anchor_vector'] is None
    
    def test_compute_meta_vector_medoid(self):
        """Test meta vector computation with medoid method"""
        result = compute_meta_vector(self.test_vectors, method="medoid")
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "medoid"
        assert isinstance(result['meta_vector'], np.ndarray)
        assert result['meta_vector'].shape == (768,)
    
    def test_compute_meta_vector_weighted_mean(self):
        """Test meta vector computation with weighted mean method"""
        result = compute_meta_vector(self.test_vectors, method="weighted_mean", weights=self.test_weights)
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "weighted_mean"
        assert isinstance(result['meta_vector'], np.ndarray)
        assert result['meta_vector'].shape == (768,)
    
    def test_compute_meta_vector_weighted_mean_no_weights(self):
        """Test meta vector computation with weighted mean but no weights"""
        with pytest.raises(ValueError, match="Weights required for weighted_mean method"):
            compute_meta_vector(self.test_vectors, method="weighted_mean")
    
    def test_compute_meta_vector_geometric_median(self):
        """Test meta vector computation with geometric median method"""
        result = compute_meta_vector(self.test_vectors, method="geometric_median")
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "geometric_median"
        assert isinstance(result['meta_vector'], np.ndarray)
        assert result['meta_vector'].shape == (768,)
    
    @patch('core.meta_vector.PCA')
    def test_compute_meta_vector_first_principal_component(self, mock_pca):
        """Test meta vector computation with first principal component method"""
        # Mock PCA
        mock_pca_instance = MagicMock()
        mock_pca_instance.components_ = np.random.randn(1, 768)
        mock_pca_instance.components_[0] = mock_pca_instance.components_[0] / np.linalg.norm(mock_pca_instance.components_[0])
        mock_pca.return_value = mock_pca_instance
        
        result = compute_meta_vector(self.test_vectors, method="first_principal_component")
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "first_principal_component"
        assert isinstance(result['meta_vector'], np.ndarray)
        assert result['meta_vector'].shape == (768,)
    
    def test_compute_meta_vector_unknown_method(self):
        """Test meta vector computation with unknown method"""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_meta_vector(self.test_vectors, method="unknown_method")
    
    def test_compute_meta_vector_with_anchor_idx(self):
        """Test meta vector computation with anchor index"""
        result = compute_meta_vector(self.test_vectors, method="centroid", anchor_idx=0)
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'anchor_vector' in result
        assert result['anchor_vector'] is not None
        assert np.allclose(result['anchor_vector'], self.test_vectors[0])
    
    def test_compute_meta_vector_invalid_anchor_idx(self):
        """Test meta vector computation with invalid anchor index"""
        result = compute_meta_vector(self.test_vectors, method="centroid", anchor_idx=999)
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert result['anchor_vector'] is None  # Should handle gracefully
    
    def test_compute_meta_vector_default_method(self):
        """Test meta vector computation with default method (centroid)"""
        result = compute_meta_vector(self.test_vectors)
        
        assert isinstance(result, dict)
        assert 'meta_vector' in result
        assert 'method' in result
        assert result['method'] == "centroid"
    
    def test_integration_all_methods(self):
        """Integration test: compare all methods on same data"""
        methods = ["centroid", "medoid", "geometric_median"]
        results = {}
        
        for method in methods:
            if method == "weighted_mean":
                results[method] = compute_meta_vector(self.test_vectors, method=method, weights=self.test_weights)
            else:
                results[method] = compute_meta_vector(self.test_vectors, method=method)
        
        # All should return valid vectors
        for method, result in results.items():
            assert isinstance(result['meta_vector'], np.ndarray)
            assert result['meta_vector'].shape == (768,)
            assert np.linalg.norm(result['meta_vector']) > 0
            assert result['method'] == method 