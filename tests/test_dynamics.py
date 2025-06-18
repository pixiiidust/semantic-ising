import pytest
import numpy as np
from unittest.mock import patch

# Import the functions we'll implement
from core.dynamics import compute_correlation_matrix, compute_correlation_length, alignment_curvature


class TestDynamics:
    """Test dynamics and correlation analysis functionality"""
    
    def test_compute_correlation_matrix_basic(self):
        """Test basic correlation matrix computation"""
        # Create test vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],  # Unit vector along x-axis
            [0.0, 1.0, 0.0],  # Unit vector along y-axis
            [0.0, 0.0, 1.0]   # Unit vector along z-axis
        ])
        
        result = compute_correlation_matrix(vectors)
        
        # Check shape
        assert result.shape == (3, 3)
        
        # Check diagonal elements (self-correlation)
        assert np.allclose(np.diag(result), 0.0)
        
        # Check that orthogonal vectors have zero correlation
        assert np.allclose(result[0, 1], 0.0)
        assert np.allclose(result[0, 2], 0.0)
        assert np.allclose(result[1, 2], 0.0)
        
        # Check symmetry
        assert np.allclose(result, result.T)
    
    def test_compute_correlation_matrix_identical_vectors(self):
        """Test correlation matrix with identical vectors"""
        # Create identical vectors
        base_vector = np.array([1.0, 2.0, 3.0])
        vectors = np.array([base_vector, base_vector, base_vector])
        
        result = compute_correlation_matrix(vectors)
        
        # Check shape
        assert result.shape == (3, 3)
        
        # Check diagonal elements (self-correlation)
        assert np.allclose(np.diag(result), 0.0)
        
        # Check that identical vectors have high correlation
        # (should be close to 1.0 after normalization)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert result[i, j] > 0.9
    
    def test_compute_correlation_matrix_single_vector(self):
        """Test correlation matrix with single vector"""
        vectors = np.array([[1.0, 0.0, 0.0]])
        
        result = compute_correlation_matrix(vectors)
        
        # Should return 1x1 matrix with zero (no self-correlation)
        assert result.shape == (1, 1)
        assert np.allclose(result[0, 0], 0.0)
    
    def test_compute_correlation_matrix_empty_input(self):
        """Test correlation matrix with empty input"""
        vectors = np.array([]).reshape(0, 3)
        
        with pytest.raises(ValueError, match="Empty vectors array"):
            compute_correlation_matrix(vectors)
    
    def test_compute_correlation_length_basic(self):
        """Test basic correlation length computation"""
        # Create correlated vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to first vector
            [0.8, 0.2, 0.0],  # Similar to first vector
            [0.0, 1.0, 0.0],  # Different direction
            [0.0, 0.9, 0.1]   # Similar to fourth vector
        ])
        
        result = compute_correlation_length(vectors)
        
        # Should return a positive finite value
        assert np.isfinite(result)
        assert result > 0.0
    
    def test_compute_correlation_length_with_distance_matrix(self):
        """Test correlation length with linguistic distance matrix"""
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ])
        
        # Create distance matrix (e.g., linguistic distances)
        lang_dist_matrix = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0]
        ])
        
        result = compute_correlation_length(vectors, lang_dist_matrix)
        
        # Should return a positive finite value
        assert np.isfinite(result)
        assert result > 0.0
    
    def test_compute_correlation_length_insufficient_data(self):
        """Test correlation length with insufficient data"""
        # Only 2 vectors - insufficient for correlation analysis
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        result = compute_correlation_length(vectors)
        
        # Should return NaN for insufficient data
        assert np.isnan(result)
    
    def test_compute_correlation_length_curve_fit_failure(self):
        """Test correlation length when curve fitting fails"""
        # Create vectors that will cause curve fitting to fail
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]  # Orthogonal vectors - no clear correlation pattern
        ])
        
        result = compute_correlation_length(vectors)
        
        # Should return NaN when curve fitting fails
        assert np.isnan(result)
    
    def test_compute_correlation_length_identical_vectors(self):
        """Test correlation length with identical vectors"""
        base_vector = np.array([1.0, 2.0, 3.0])
        vectors = np.array([base_vector, base_vector, base_vector, base_vector])
        
        result = compute_correlation_length(vectors)
        
        # Should return a finite value (very high correlation)
        assert np.isfinite(result)
        assert result > 0.0
    
    def test_alignment_curvature_basic(self):
        """Test basic alignment curvature computation"""
        # Create synthetic alignment curve with known curvature
        temperatures = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        alignment_curve = np.array([0.1, 0.3, 0.5, 0.3, 0.1])  # Peak at T=1.5
        
        result = alignment_curvature(alignment_curve, temperatures)
        
        # Check shape
        assert result.shape == alignment_curve.shape
        
        # Check that curvature is negative at the peak (concave down)
        assert result[2] < 0.0  # Peak at index 2
        
        # Check that curvature is positive at the valleys (concave up)
        # Note: Endpoints are set to 0.0 by design (no neighbor on one side)
        assert result[0] == 0.0  # Endpoint at index 0
        assert result[4] == 0.0  # Endpoint at index 4
        
        # Check interior points have expected curvature
        # For this specific data: [0.1, 0.3, 0.5, 0.3, 0.1]
        # d²M/dT² at index 1: (0.5 - 2*0.3 + 0.1) / (1.5 - 0.5)² = (0.5 - 0.6 + 0.1) / 1 = 0
        # d²M/dT² at index 3: (0.1 - 2*0.3 + 0.5) / (2.5 - 1.5)² = (0.1 - 0.6 + 0.5) / 1 = 0
        # Both interior points have zero curvature due to symmetric data
        assert np.allclose(result[1], 0.0)  # Valley at index 1 (symmetric data)
        assert np.allclose(result[3], 0.0)  # Valley at index 3 (symmetric data)
    
    def test_alignment_curvature_insufficient_points(self):
        """Test alignment curvature with insufficient data points"""
        temperatures = np.array([1.0, 2.0])
        alignment_curve = np.array([0.5, 0.6])
        
        result = alignment_curvature(alignment_curve, temperatures)
        
        # Should return empty array for insufficient points
        assert len(result) == 0
    
    def test_alignment_curvature_constant_function(self):
        """Test alignment curvature with constant alignment"""
        temperatures = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        alignment_curve = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Constant
        
        result = alignment_curvature(alignment_curve, temperatures)
        
        # Curvature should be zero for constant function
        assert np.allclose(result[1:-1], 0.0)  # Interior points
    
    def test_alignment_curvature_linear_function(self):
        """Test alignment curvature with linear alignment"""
        temperatures = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        alignment_curve = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Linear
        
        result = alignment_curvature(alignment_curve, temperatures)
        
        # Curvature should be zero for linear function
        assert np.allclose(result[1:-1], 0.0)  # Interior points
    
    def test_alignment_curvature_uneven_spacing(self):
        """Test alignment curvature with uneven temperature spacing"""
        temperatures = np.array([0.5, 1.0, 2.0, 3.5, 5.0])  # Uneven spacing
        alignment_curve = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        
        result = alignment_curvature(alignment_curve, temperatures)
        
        # Should handle uneven spacing correctly
        assert result.shape == alignment_curve.shape
        assert np.isfinite(result).all()
    
    def test_alignment_curvature_edge_cases(self):
        """Test alignment curvature edge cases"""
        # Single point
        temperatures = np.array([1.0])
        alignment_curve = np.array([0.5])
        
        result = alignment_curvature(alignment_curve, temperatures)
        assert len(result) == 0
        
        # Two points
        temperatures = np.array([1.0, 2.0])
        alignment_curve = np.array([0.5, 0.6])
        
        result = alignment_curvature(alignment_curve, temperatures)
        assert len(result) == 0
        
        # Three points (minimum for curvature)
        temperatures = np.array([1.0, 2.0, 3.0])
        alignment_curve = np.array([0.1, 0.5, 0.1])
        
        result = alignment_curvature(alignment_curve, temperatures)
        assert len(result) == 3
        assert result[1] < 0.0  # Negative curvature at peak 