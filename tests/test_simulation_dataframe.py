"""
Tests to ensure simulation always returns non-empty dataframes with required columns.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import run_temperature_sweep
from core.temperature_estimation import estimate_practical_range


def dummy_vectors(n=10, dim=5):
    """Create dummy normalized vectors for testing."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_dataframe_non_empty_and_columns():
    """Test that simulation returns non-empty dataframe with required columns."""
    vecs = dummy_vectors()
    
    # Estimate temperature range
    tmin, tmax = estimate_practical_range(vecs)
    
    # Run simulation with fast parameters for testing
    sim_params = {
        'max_iterations': 100,  # Keep ultra-fast for CI
        'convergence_threshold': 0.5,  # Loose tolerance so it converges
        'noise_sigma': 0.1,
        'steps_per_T': 100
    }
    
    result = run_temperature_sweep(
        vectors=vecs,
        T_range=np.linspace(tmin, tmax, 5),  # Small number of temperatures
        store_all_temperatures=False,
        n_sweeps_per_temperature=1,  # Single sweep for speed
        sim_params=sim_params
    )
    
    # Check required columns exist
    required_columns = {"temperatures", "alignment", "entropy", "energy", "correlation_length"}
    assert required_columns.issubset(set(result.keys())), f"Missing columns: {required_columns - set(result.keys())}"
    
    # Check dataframe is non-empty
    assert len(result['temperatures']) > 0, "Temperature array should not be empty"
    assert len(result['alignment']) > 0, "Alignment array should not be empty"
    
    # Check all arrays have same length
    array_lengths = [len(result[key]) for key in required_columns]
    assert len(set(array_lengths)) == 1, f"All arrays should have same length, got: {array_lengths}"
    
    # Check that temperatures are in expected range
    temperatures = result['temperatures']
    assert np.all(temperatures >= 0), "All temperatures should be non-negative"
    assert np.all(temperatures <= 15.0), "All temperatures should be reasonable"


def test_simulation_handles_diverging_temperatures():
    """Test that simulation handles diverging temperatures gracefully."""
    vecs = dummy_vectors()
    
    # Use very low temperatures that might cause divergence
    T_range = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    
    sim_params = {
        'max_iterations': 50,  # Very fast for testing
        'convergence_threshold': 0.1,
        'noise_sigma': 0.05,
        'steps_per_T': 50
    }
    
    result = run_temperature_sweep(
        vectors=vecs,
        T_range=T_range,
        store_all_temperatures=False,
        n_sweeps_per_temperature=1,
        sim_params=sim_params
    )
    
    # Should still return results for all temperatures
    assert len(result['temperatures']) == len(T_range)
    assert len(result['alignment']) == len(T_range)
    
    # Some temperatures might have NaN values (diverging), but structure should be intact
    assert 'convergence_data' in result
    assert len(result['convergence_data']) == len(T_range)


def test_simulation_with_string_parameters():
    """Test that simulation works with string parameters (type conversion)."""
    vecs = dummy_vectors()
    
    # Use string parameters to test type conversion
    sim_params = {
        'max_iterations': "100",
        'convergence_threshold': "0.5",
        'noise_sigma': "0.1",
        'steps_per_T': "100"
    }
    
    result = run_temperature_sweep(
        vectors=vecs,
        T_range=np.array([0.5, 1.0, 1.5]),
        store_all_temperatures=False,
        n_sweeps_per_temperature=1,
        sim_params=sim_params
    )
    
    # Should work without type errors
    assert len(result['temperatures']) == 3
    assert len(result['alignment']) == 3
    
    # Check that results are numeric
    assert all(isinstance(x, (int, float, np.number)) for x in result['temperatures'])
    assert all(isinstance(x, (int, float, np.number)) or np.isnan(x) for x in result['alignment'])


def test_simulation_edge_cases():
    """Test simulation with edge cases."""
    # Test with single vector
    single_vec = dummy_vectors(n=1)
    
    result = run_temperature_sweep(
        vectors=single_vec,
        T_range=np.array([1.0, 1.5]),  # At least 2 points required
        store_all_temperatures=False,
        n_sweeps_per_temperature=1,
        sim_params={'max_iterations': 10, 'convergence_threshold': 0.1}
    )
    
    assert len(result['temperatures']) == 2
    assert len(result['alignment']) == 2
    
    # Test with identical vectors
    identical_vecs = np.tile(dummy_vectors(n=1), (3, 1))
    
    result = run_temperature_sweep(
        vectors=identical_vecs,
        T_range=np.array([1.0, 1.5]),  # At least 2 points required
        store_all_temperatures=False,
        n_sweeps_per_temperature=1,
        sim_params={'max_iterations': 10, 'convergence_threshold': 0.1}
    )
    
    assert len(result['temperatures']) == 2
    assert len(result['alignment']) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 