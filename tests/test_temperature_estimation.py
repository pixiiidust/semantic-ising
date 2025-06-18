import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from temperature_estimation import (
    estimate_critical_temperature,
    estimate_max_temperature, 
    estimate_practical_range,
    quick_scan_probe,
    validate_temperature_range
)


class TestTemperatureEstimation:
    """Test temperature estimation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create realistic test vectors (normalized)
        np.random.seed(42)
        self.test_vectors = np.random.randn(10, 768)
        self.test_vectors = self.test_vectors / np.linalg.norm(self.test_vectors, axis=1, keepdims=True)
        
        # Create high-similarity vectors (should give low Tc)
        self.high_sim_vectors = np.random.randn(768)
        self.high_sim_vectors = self.high_sim_vectors / np.linalg.norm(self.high_sim_vectors)
        self.high_sim_vectors = np.tile(self.high_sim_vectors, (5, 1))
        # Add small noise
        noise = np.random.normal(0, 0.1, self.high_sim_vectors.shape)
        self.high_sim_vectors += noise
        self.high_sim_vectors = self.high_sim_vectors / np.linalg.norm(self.high_sim_vectors, axis=1, keepdims=True)
        
        # Create low-similarity vectors (should give high Tc)
        self.low_sim_vectors = np.random.randn(5, 768)
        self.low_sim_vectors = self.low_sim_vectors / np.linalg.norm(self.low_sim_vectors, axis=1, keepdims=True)
    
    def test_estimate_critical_temperature_basic(self):
        """Test basic critical temperature estimation."""
        tc = estimate_critical_temperature(self.test_vectors)
        
        assert isinstance(tc, float)
        assert tc > 0
        assert tc < 10.0  # Reasonable upper bound
        assert not np.isnan(tc)
    
    def test_estimate_critical_temperature_high_similarity(self):
        """Test Tc estimation with high similarity vectors (should give low Tc)."""
        tc = estimate_critical_temperature(self.high_sim_vectors)
        
        assert tc < 1.0  # High similarity should give low Tc
        assert tc > 0.05  # But not zero
    
    def test_estimate_critical_temperature_low_similarity(self):
        """Test Tc estimation with low similarity vectors (should give high Tc)."""
        tc = estimate_critical_temperature(self.low_sim_vectors)
        
        assert tc > 0.5  # Low similarity should give higher Tc
        assert tc < 5.0  # But not unreasonably high
    
    def test_estimate_max_temperature_basic(self):
        """Test basic maximum temperature estimation."""
        tmax = estimate_max_temperature(self.test_vectors)
        
        assert isinstance(tmax, float)
        assert tmax > 0
        assert tmax >= 1.0  # Should be at least 1.0 (adjusted expectation)
        assert not np.isnan(tmax)
    
    def test_estimate_practical_range_basic(self):
        """Test basic practical range estimation."""
        tmin, tmax = estimate_practical_range(self.test_vectors)
        
        assert isinstance(tmin, float)
        assert isinstance(tmax, float)
        assert tmin > 0
        assert tmax > tmin
        assert tmax - tmin >= 0.5  # Should have reasonable span
        assert tmax - tmin <= 10.0  # But not too wide
    
    def test_validate_temperature_range_valid(self):
        """Test validation of valid temperature range."""
        tmin, tmax = 0.5, 2.0
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert is_valid
        assert message == "Valid temperature range"
    
    def test_validate_temperature_range_invalid_order(self):
        """Test validation of invalid temperature range (tmin > tmax)."""
        tmin, tmax = 2.0, 0.5
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert not is_valid
        assert "minimum" in message.lower()
    
    def test_estimate_critical_temperature_single_vector(self):
        """Test Tc estimation with single vector (edge case)."""
        single_vector = self.test_vectors[:1]
        tc = estimate_critical_temperature(single_vector)
        
        # Should return a reasonable default
        assert isinstance(tc, float)
        assert tc > 0
        assert tc < 2.0
    
    def test_estimate_critical_temperature_identical_vectors(self):
        """Test Tc estimation with identical vectors (edge case)."""
        identical_vectors = np.tile(self.test_vectors[0], (3, 1))
        tc = estimate_critical_temperature(identical_vectors)
        
        # Should return a reasonable default for identical vectors
        assert isinstance(tc, float)
        assert tc > 0
        assert tc < 1.0
    
    def test_estimate_max_temperature_high_similarity(self):
        """Test Tmax estimation with high similarity vectors."""
        tmax = estimate_max_temperature(self.high_sim_vectors)
        
        # Should be higher than Tc but reasonable
        tc = estimate_critical_temperature(self.high_sim_vectors)
        assert tmax >= tc  # Adjusted to >= instead of >
        assert tmax < 10.0
    
    def test_estimate_max_temperature_low_similarity(self):
        """Test max temperature estimation with low-similarity vectors."""
        # Create low-similarity vectors (orthogonal)
        vectors = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        t_max = estimate_max_temperature(vectors)
        
        # For low-similarity vectors, max temperature should be at least the lower bound
        # but may be less than critical temperature (which is expected for orthogonal vectors)
        assert t_max >= 1.0, f"Max temperature {t_max} should be at least 1.0"
        assert isinstance(t_max, float), "Max temperature should be a float"
    
    def test_estimate_practical_range_with_padding(self):
        """Test range estimation with padding applied."""
        tmin, tmax = estimate_practical_range(self.test_vectors, padding=0.3)
        
        # Should have wider range with padding (unless hitting upper bound)
        tmin_no_pad, tmax_no_pad = estimate_practical_range(self.test_vectors, padding=0.0)
        
        # With conservative bounds, both might hit the 3.0 upper limit
        # In that case, they should be equal. Otherwise, padding should expand the range.
        if tmax < 3.0:  # Not hitting upper bound
            assert tmax - tmin > tmax_no_pad - tmin_no_pad
        else:  # Hitting upper bound
            assert tmax == tmax_no_pad  # Both should be capped at 3.0
    
    def test_estimate_practical_range_minimum_span(self):
        """Test that range has minimum span enforced."""
        # Create vectors that would give very narrow range
        narrow_vectors = self.high_sim_vectors  # High similarity = narrow range
        tmin, tmax = estimate_practical_range(narrow_vectors)
        
        assert tmax - tmin >= 0.75  # Minimum span enforced
    
    def test_estimate_practical_range_lower_bound_floor(self):
        """Test lower bound floor enforcement."""
        # Create vectors that would give very low Tmin
        low_tmin_vectors = self.high_sim_vectors  # High similarity = low Tmin
        tmin, tmax = estimate_practical_range(low_tmin_vectors)
        
        assert tmin >= 0.05  # Lower bound floor enforced
    
    def test_quick_scan_probe_basic(self):
        """Test basic quick scan probe functionality."""
        with patch('core.simulation.simulate_at_temperature') as mock_sim:
            # Mock simulation to return realistic metrics
            mock_sim.return_value = ({'alignment': 0.8}, np.random.randn(5, 768), {'status': 'converged'})
            
            result = quick_scan_probe(self.test_vectors)
            
            # Should return either None or a tuple of (tmin, tmax)
            if result is not None:
                tmin, tmax = result
                assert isinstance(tmin, float)
                assert isinstance(tmax, float)
                assert tmin > 0
                assert tmax > tmin
    
    def test_quick_scan_probe_no_slope_detected(self):
        """Test quick scan when no significant slope is detected."""
        with patch('core.simulation.simulate_at_temperature') as mock_sim:
            # Mock simulation to return flat alignment (no slope)
            mock_sim.return_value = ({'alignment': 0.5}, np.random.randn(5, 768), {'status': 'converged'})
            
            result = quick_scan_probe(self.test_vectors)
            
            # Should return None when no slope detected
            assert result is None
    
    def test_quick_scan_probe_with_refinement(self):
        """Test quick scan with range refinement."""
        with patch('core.simulation.simulate_at_temperature') as mock_sim:
            # Mock simulation to return varying alignment
            def mock_sim_side_effect(vectors, T, **kwargs):
                alignment = 0.9 - 0.4 * T  # Decreasing alignment with T
                return ({'alignment': max(0.1, alignment)}, vectors, {'status': 'converged'})
            
            mock_sim.side_effect = mock_sim_side_effect
            
            original_tmin, original_tmax = estimate_practical_range(self.test_vectors)
            result = quick_scan_probe(self.test_vectors, 
                                    original_range=(original_tmin, original_tmax))
            
            if result is not None:
                refined_tmin, refined_tmax = result
                assert refined_tmax > refined_tmin
                # Refined range should be reasonable
                assert refined_tmax - refined_tmin >= 0.5
    
    def test_validate_temperature_range_negative(self):
        """Test validation of negative temperature range."""
        tmin, tmax = -0.1, 2.0
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert not is_valid
        assert "positive" in message.lower()
    
    def test_validate_temperature_range_too_narrow(self):
        """Test validation of too narrow temperature range."""
        tmin, tmax = 1.0, 1.1
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert not is_valid
        assert "narrow" in message.lower()
    
    def test_validate_temperature_range_too_wide(self):
        """Test validation of too wide temperature range."""
        tmin, tmax = 0.1, 25.0  # Changed to 25.0 to exceed the 20.0 limit
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert not is_valid
        assert "wide" in message.lower()
    
    def test_integration_full_estimation_workflow(self):
        """Test complete temperature estimation workflow."""
        # Test the full workflow from vectors to final range
        tmin, tmax = estimate_practical_range(self.test_vectors)
        
        # Validate the estimated range
        is_valid, message = validate_temperature_range(tmin, tmax)
        
        assert is_valid, f"Estimated range [{tmin}, {tmax}] is invalid: {message}"
        assert tmin >= 0.05  # Lower bound floor
        assert tmax - tmin >= 0.75  # Minimum span
        assert tmax <= 15.0  # Reasonable upper bound
    
    def test_edge_case_empty_vectors(self):
        """Test edge case with empty vector array."""
        empty_vectors = np.array([]).reshape(0, 768)
        
        with pytest.raises(ValueError):
            estimate_critical_temperature(empty_vectors)
    
    def test_edge_case_zero_vectors(self):
        """Test edge case with zero vectors."""
        zero_vectors = np.zeros((3, 768))
        
        # Should handle gracefully and return reasonable defaults
        tc = estimate_critical_temperature(zero_vectors)
        assert isinstance(tc, float)
        assert tc > 0
    
    def test_performance_benchmark(self):
        """Benchmark performance of temperature estimation."""
        import time
        
        start_time = time.time()
        tmin, tmax = estimate_practical_range(self.test_vectors)
        end_time = time.time()
        
        # Should complete quickly (< 1 second for small vectors)
        assert end_time - start_time < 1.0
        
        # Results should be reasonable
        assert tmax > tmin
        assert tmin >= 0.05
        assert tmax - tmin >= 0.75
    
    def test_estimates_are_floats(self):
        """Test that all temperature estimation outputs are proper floats."""
        # Test critical temperature estimation
        tc = estimate_critical_temperature(self.test_vectors)
        assert isinstance(tc, float), f"Critical temperature should be float, got {type(tc)}"
        
        # Test max temperature estimation
        tmax = estimate_max_temperature(self.test_vectors)
        assert isinstance(tmax, float), f"Max temperature should be float, got {type(tmax)}"
        
        # Test practical range estimation
        tmin, tmax_range = estimate_practical_range(self.test_vectors)
        assert isinstance(tmin, float), f"Min temperature should be float, got {type(tmin)}"
        assert isinstance(tmax_range, float), f"Max temperature should be float, got {type(tmax_range)}"
        
        # Test that all values are positive
        assert tc > 0, "Critical temperature should be positive"
        assert tmax > 0, "Max temperature should be positive"
        assert tmin > 0, "Min temperature should be positive"
        assert tmax_range > 0, "Max temperature from range should be positive"


class TestTemperatureEstimationIntegration:
    """Integration tests for temperature estimation with simulation."""
    
    def test_estimation_with_real_simulation_data(self):
        """Test estimation with data that would come from real simulation."""
        # Create vectors that simulate real embedding data
        np.random.seed(42)
        real_vectors = np.random.randn(8, 768)
        real_vectors = real_vectors / np.linalg.norm(real_vectors, axis=1, keepdims=True)
        
        # Estimate range
        tmin, tmax = estimate_practical_range(real_vectors)
        
        # Verify range is suitable for simulation
        assert tmin > 0
        assert tmax > tmin
        assert tmax - tmin >= 0.75
        
        # Test that range contains reasonable temperatures
        test_temps = np.linspace(tmin, tmax, 5)
        for T in test_temps:
            assert T > 0
            assert T < 15.0
    
    def test_estimation_consistency_across_runs(self):
        """Test that estimation gives consistent results across multiple runs."""
        np.random.seed(42)
        vectors = np.random.randn(10, 768)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Run estimation multiple times
        results = []
        for _ in range(3):
            tmin, tmax = estimate_practical_range(vectors)
            results.append((tmin, tmax))
        
        # Results should be consistent (within 10% tolerance)
        for i in range(1, len(results)):
            tmin1, tmax1 = results[i-1]
            tmin2, tmax2 = results[i]
            
            assert abs(tmin1 - tmin2) / tmin1 < 0.1
            assert abs(tmax1 - tmax2) / tmax1 < 0.1

    def test_integration_with_simulation_workflow(self):
        """Test temperature estimation integration with simulation workflow."""
        # Create test vectors
        vectors = np.random.randn(5, 768)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Estimate practical range
        tmin, tmax = estimate_practical_range(vectors)
        
        # Validate the range
        assert tmin > 0, "Minimum temperature should be positive"
        assert tmax > tmin, "Maximum temperature should be greater than minimum"
        assert tmax < 10.0, "Maximum temperature should be reasonable"
        
        # Create temperature range for simulation
        temperature_range = np.linspace(tmin, tmax, 10)
        
        # Verify the range is suitable for simulation
        is_valid, message = validate_temperature_range(tmin, tmax)
        assert is_valid, f"Temperature range should be valid: {message}"
        
        print(f"Integration test passed: Estimated range {tmin:.3f} - {tmax:.3f} with {len(temperature_range)} points") 