"""
Tests for post-analysis validation to ensure proper handling of invalid data.
"""

import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.post_analysis import validate_analysis_inputs, clean_simulation_results


def test_validate_drops_invalid_and_passes():
    """Test that validation drops invalid rows and passes valid ones."""
    # Create simulation results with mixed valid/invalid data (need at least 3 valid points)
    simulation_results = {
        'temperatures': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'alignment': np.array([0.9, np.nan, 0.7, 0.6, np.nan]),  # 3 valid, 2 invalid
        'entropy': np.array([0.1, np.nan, 0.3, 0.4, np.nan]),
        'energy': np.array([-0.8, np.nan, -0.6, -0.5, np.nan]),
        'correlation_length': np.array([1.2, np.nan, 1.0, 0.9, np.nan]),
        'convergence_data': [
            {'status': 'converged', 'final_diff': 1e-6},
            {'status': 'diverging', 'final_diff': 0.1},
            {'status': 'converged', 'final_diff': 1e-5},
            {'status': 'converged', 'final_diff': 1e-4},
            {'status': 'diverging', 'final_diff': 0.2}
        ]
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Test validation (should pass with 3 valid points)
    is_valid = validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.15)
    assert is_valid is True
    
    # Test cleaning
    clean_results = clean_simulation_results(simulation_results)
    
    # Should have 3 valid rows (rows 0, 2, 3)
    assert len(clean_results['temperatures']) == 3
    assert clean_results['temperatures'][0] == 0.1
    assert clean_results['temperatures'][1] == 0.3
    assert clean_results['temperatures'][2] == 0.4
    assert clean_results['alignment'][0] == 0.9
    assert clean_results['alignment'][1] == 0.7
    assert clean_results['alignment'][2] == 0.6
    
    # Check that no NaN values remain
    assert not np.any(np.isnan(clean_results['alignment']))
    assert not np.any(np.isnan(clean_results['entropy']))
    assert not np.any(np.isnan(clean_results['energy']))
    assert not np.any(np.isnan(clean_results['correlation_length']))


def test_validate_raises_when_all_invalid():
    """Test that validation raises error when all data is invalid."""
    # Create simulation results with all invalid data
    simulation_results = {
        'temperatures': np.array([0.3, 0.4]),
        'alignment': np.array([np.nan, np.nan]),
        'entropy': np.array([np.nan, np.nan]),
        'energy': np.array([np.nan, np.nan]),
        'correlation_length': np.array([np.nan, np.nan]),
        'convergence_data': [
            {'status': 'diverging', 'final_diff': 0.1},
            {'status': 'diverging', 'final_diff': 0.2}
        ]
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Should raise ValueError when all data is invalid
    with pytest.raises(ValueError, match="All temperature points were diverging or invalid"):
        validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.2)


def test_validate_handles_empty_results():
    """Test validation with empty simulation results."""
    # Create empty simulation results
    simulation_results = {
        'temperatures': np.array([]),
        'alignment': np.array([]),
        'entropy': np.array([]),
        'energy': np.array([]),
        'correlation_length': np.array([]),
        'convergence_data': []
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Should raise error for empty results
    with pytest.raises(ValueError, match="All temperature points were diverging or invalid"):
        validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.2)


def test_validate_handles_missing_columns():
    """Test validation with missing columns in simulation results."""
    # Create simulation results missing some columns (need at least 3 valid points)
    simulation_results = {
        'temperatures': np.array([0.1, 0.2, 0.3, 0.4]),
        'alignment': np.array([0.9, 0.8, 0.7, 0.6]),
        # Missing entropy, energy, correlation_length
        'convergence_data': [
            {'status': 'converged', 'final_diff': 1e-6},
            {'status': 'converged', 'final_diff': 1e-5},
            {'status': 'converged', 'final_diff': 1e-4},
            {'status': 'converged', 'final_diff': 1e-3}
        ]
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Should handle missing columns gracefully
    is_valid = validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.15)
    assert is_valid is True
    
    # Test cleaning
    clean_results = clean_simulation_results(simulation_results)
    
    # Should preserve valid data
    assert len(clean_results['temperatures']) == 4
    assert len(clean_results['alignment']) == 4
    assert clean_results['temperatures'][0] == 0.1
    assert clean_results['temperatures'][1] == 0.2


def test_validate_handles_partial_nan():
    """Test validation with partial NaN values."""
    # Create simulation results with some NaN values (need at least 3 valid points)
    simulation_results = {
        'temperatures': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        'alignment': np.array([0.9, np.nan, 0.7, np.nan, 0.5, 0.4]),
        'entropy': np.array([0.1, 0.2, np.nan, 0.4, np.nan, 0.6]),
        'energy': np.array([-0.8, -0.7, -0.6, np.nan, -0.4, np.nan]),
        'correlation_length': np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7]),
        'convergence_data': [
            {'status': 'converged', 'final_diff': 1e-6},
            {'status': 'diverging', 'final_diff': 0.1},
            {'status': 'converged', 'final_diff': 1e-5},
            {'status': 'diverging', 'final_diff': 0.2},
            {'status': 'converged', 'final_diff': 1e-4},
            {'status': 'diverging', 'final_diff': 0.3}
        ]
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Test validation (should pass with 3 valid points)
    is_valid = validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.25)
    assert is_valid is True
    
    # Test cleaning
    clean_results = clean_simulation_results(simulation_results)
    
    # Only the first row is fully finite across all metrics
    assert len(clean_results['temperatures']) == 1
    assert clean_results['temperatures'][0] == 0.1
    assert clean_results['alignment'][0] == 0.9
    
    # Check that no NaN values remain in any column
    for key in ['alignment', 'entropy', 'energy', 'correlation_length']:
        if key in clean_results:
            assert not np.any(np.isnan(clean_results[key]))


def test_validate_preserves_convergence_data():
    """Test that validation preserves convergence data for valid rows."""
    # Create simulation results (need at least 3 valid points)
    simulation_results = {
        'temperatures': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'alignment': np.array([0.9, np.nan, 0.7, 0.6, np.nan]),
        'entropy': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'energy': np.array([-0.8, -0.7, -0.6, -0.5, -0.4]),
        'correlation_length': np.array([1.2, 1.1, 1.0, 0.9, 0.8]),
        'convergence_data': [
            {'status': 'converged', 'final_diff': 1e-6, 'iterations': 100},
            {'status': 'diverging', 'final_diff': 0.1, 'iterations': 50},
            {'status': 'converged', 'final_diff': 1e-5, 'iterations': 200},
            {'status': 'converged', 'final_diff': 1e-4, 'iterations': 150},
            {'status': 'diverging', 'final_diff': 0.2, 'iterations': 75}
        ]
    }
    
    # Create anchor vectors
    anchor_vectors = np.random.randn(2, 768)
    anchor_vectors /= np.linalg.norm(anchor_vectors, axis=1, keepdims=True)
    
    # Test validation (should pass with 3 valid points)
    is_valid = validate_analysis_inputs(simulation_results, anchor_vectors, tc=0.15)
    assert is_valid is True
    
    # Test cleaning
    clean_results = clean_simulation_results(simulation_results)
    
    # Should preserve convergence data for valid rows (indices 0, 2, 3)
    assert len(clean_results['convergence_data']) == 3
    assert clean_results['convergence_data'][0]['status'] == 'converged'
    assert clean_results['convergence_data'][1]['status'] == 'converged'
    assert clean_results['convergence_data'][2]['status'] == 'converged'
    assert clean_results['convergence_data'][0]['iterations'] == 100
    assert clean_results['convergence_data'][1]['iterations'] == 200
    assert clean_results['convergence_data'][2]['iterations'] == 150


if __name__ == "__main__":
    pytest.main([__file__]) 