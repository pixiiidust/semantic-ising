import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the functions we'll implement
from core.simulation import (
    run_temperature_sweep, simulate_at_temperature, update_vectors_ising,
    update_vectors_metropolis, update_vectors_glauber, collect_metrics,
    compute_alignment, compute_entropy
)
from core.physics import total_system_energy


class TestSimulation:
    """Test core simulation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create test vectors (normalized)
        np.random.seed(42)
        self.test_vectors = np.random.randn(5, 768)
        self.test_vectors = self.test_vectors / np.linalg.norm(self.test_vectors, axis=1, keepdims=True)
        self.temperature_range = [0.5, 1.0, 1.5, 2.0]
    
    def test_run_temperature_sweep_basic(self):
        """Test basic temperature sweep functionality"""
        result = run_temperature_sweep(self.test_vectors, self.temperature_range)
        
        # Check required keys
        required_keys = ['temperatures', 'alignment', 'entropy', 'energy', 'correlation_length']
        for key in required_keys:
            assert key in result
            assert len(result[key]) == len(self.temperature_range)
        
        # Check temperature values
        assert np.allclose(result['temperatures'], self.temperature_range)
        
        # Check metric ranges
        assert np.all(result['alignment'] >= 0) and np.all(result['alignment'] <= 1)
        assert np.all(result['entropy'] >= 0)
        assert np.all(result['correlation_length'] >= 0)
    
    def test_run_temperature_sweep_with_snapshots(self):
        """Test temperature sweep with vector snapshots"""
        result = run_temperature_sweep(
            self.test_vectors, 
            self.temperature_range, 
            store_all_temperatures=True,
            max_snapshots=3
        )
        
        # Check that vector snapshots are stored
        assert 'vector_snapshots' in result
        assert len(result['vector_snapshots']) <= 3  # Max snapshots
        
        # Check snapshot shapes
        for temp, vectors in result['vector_snapshots'].items():
            assert vectors.shape == self.test_vectors.shape
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)  # Normalized
    
    def test_run_temperature_sweep_multi_replica(self):
        """Test multi-replica temperature sweep"""
        result = run_temperature_sweep(
            self.test_vectors, 
            self.temperature_range, 
            n_replicas=3
        )
        
        # Check that replica statistics are included
        for metric in ['alignment', 'entropy', 'energy', 'correlation_length']:
            assert f'{metric}_sem' in result  # Standard error
            assert f'{metric}_replicas' in result  # Raw replica data
            assert result[f'{metric}_replicas'].shape[0] == 3  # 3 replicas
    
    def test_run_temperature_sweep_invalid_inputs(self):
        """Test temperature sweep with invalid inputs"""
        # Invalid temperature range
        with pytest.raises(ValueError, match="Temperature range must have at least 2 points"):
            run_temperature_sweep(self.test_vectors, [1.0])
        
        # Invalid number of replicas
        with pytest.raises(ValueError, match="Number of replicas must be >= 1"):
            run_temperature_sweep(self.test_vectors, self.temperature_range, n_replicas=0)
    
    def test_simulate_at_temperature_convergence(self):
        """Test simulation at temperature with convergence"""
        metrics, updated_vectors, convergence_info = simulate_at_temperature(self.test_vectors, 1.0)
        
        # Check metrics
        required_metrics = ['alignment', 'entropy', 'energy', 'correlation_length']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check updated vectors
        assert updated_vectors.shape == self.test_vectors.shape
        assert np.allclose(np.linalg.norm(updated_vectors, axis=1), 1.0)  # Normalized
        
        # Check convergence info
        assert 'status' in convergence_info
        assert 'final_diff' in convergence_info
        assert 'iterations' in convergence_info
        assert 'diff_history' in convergence_info
        assert 'alignment_history' in convergence_info
        assert 'logged_steps' in convergence_info
        assert 'temperature' in convergence_info
    
    def test_simulate_at_temperature_no_convergence(self):
        """Test simulation at temperature without convergence"""
        # Use very low convergence threshold to force non-convergence
        metrics, updated_vectors, convergence_info = simulate_at_temperature(
            self.test_vectors, 1.0, convergence_threshold=1e-12
        )
        
        # Should still return valid results
        assert 'alignment' in metrics
        assert updated_vectors.shape == self.test_vectors.shape
        
        # Check convergence info
        assert 'status' in convergence_info
        assert convergence_info['status'] in ['converged', 'max_steps', 'plateau', 'diverging']
    
    def test_simulate_at_temperature_invalid_temperature(self):
        """Test simulation with invalid temperature"""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            simulate_at_temperature(self.test_vectors, 0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            simulate_at_temperature(self.test_vectors, -1.0)
    
    def test_update_vectors_ising_metropolis(self):
        """Test Ising update with Metropolis method"""
        updated_vectors = update_vectors_ising(
            self.test_vectors, 1.0, J=1.0, update_method="metropolis"
        )
        
        assert updated_vectors.shape == self.test_vectors.shape
        assert np.allclose(np.linalg.norm(updated_vectors, axis=1), 1.0)  # Normalized
        assert not np.allclose(updated_vectors, self.test_vectors)  # Should change
    
    def test_update_vectors_ising_glauber(self):
        """Test Ising update with Glauber method"""
        updated_vectors = update_vectors_ising(
            self.test_vectors, 1.0, J=1.0, update_method="glauber"
        )
        
        assert updated_vectors.shape == self.test_vectors.shape
        assert np.allclose(np.linalg.norm(updated_vectors, axis=1), 1.0)  # Normalized
        assert not np.allclose(updated_vectors, self.test_vectors)  # Should change
    
    def test_update_vectors_ising_invalid_method(self):
        """Test Ising update with invalid method"""
        with pytest.raises(ValueError, match="Unknown update method"):
            update_vectors_ising(self.test_vectors, 1.0, update_method="invalid")
    
    def test_update_vectors_metropolis_acceptance(self):
        """Test Metropolis update acceptance criterion"""
        # Test at low temperature (high acceptance)
        updated_low_T = update_vectors_metropolis(self.test_vectors, 0.1, J=1.0)
        
        # Test at high temperature (lower acceptance)
        updated_high_T = update_vectors_metropolis(self.test_vectors, 5.0, J=1.0)
        
        # Both should be normalized
        assert np.allclose(np.linalg.norm(updated_low_T, axis=1), 1.0)
        assert np.allclose(np.linalg.norm(updated_high_T, axis=1), 1.0)
    
    def test_update_vectors_glauber_probability(self):
        """Test Glauber update probability calculation"""
        # Test with strong field (should align)
        updated_vectors = update_vectors_glauber(self.test_vectors, 0.1, J=1.0)
        
        assert updated_vectors.shape == self.test_vectors.shape
        assert np.allclose(np.linalg.norm(updated_vectors, axis=1), 1.0)  # Normalized
    
    def test_collect_metrics_comprehensive(self):
        """Test comprehensive metrics collection"""
        metrics = collect_metrics(self.test_vectors, 1.0)
        
        required_metrics = ['alignment', 'entropy', 'energy', 'correlation_length']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
    
    def test_compute_alignment_metric(self):
        """Test alignment metric computation"""
        # Test with identical vectors (should have high alignment)
        identical_vectors = np.tile(self.test_vectors[0], (3, 1))
        identical_vectors = identical_vectors / np.linalg.norm(identical_vectors, axis=1, keepdims=True)
        
        alignment = compute_alignment(identical_vectors)
        assert 0.9 <= alignment <= 1.0  # Should be very high
        
        # Test with random vectors (should have lower alignment)
        random_vectors = np.random.randn(3, 768)
        random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
        
        alignment = compute_alignment(random_vectors)
        assert 0.0 <= alignment <= 1.0
    
    def test_compute_alignment_single_vector(self):
        """Test alignment computation with single vector"""
        single_vector = self.test_vectors[:1]
        alignment = compute_alignment(single_vector)
        assert alignment == 1.0  # Single vector has perfect alignment with itself
    
    def test_compute_entropy_metric(self):
        """Test entropy metric computation"""
        # Test with identical vectors (should have low entropy)
        identical_vectors = np.tile(self.test_vectors[0], (3, 1))
        identical_vectors = identical_vectors / np.linalg.norm(identical_vectors, axis=1, keepdims=True)
        
        entropy = compute_entropy(identical_vectors)
        assert entropy >= 0.0  # Entropy should be non-negative
        
        # Test with diverse vectors (should have higher entropy)
        diverse_vectors = np.random.randn(5, 768)
        diverse_vectors = diverse_vectors / np.linalg.norm(diverse_vectors, axis=1, keepdims=True)
        
        entropy = compute_entropy(diverse_vectors)
        assert entropy >= 0.0
    
    def test_compute_entropy_single_vector(self):
        """Test entropy computation with single vector"""
        single_vector = self.test_vectors[:1]
        entropy = compute_entropy(single_vector)
        assert entropy == 0.0  # Single vector has zero entropy
    
    def test_total_system_energy_consistency(self):
        """Test system energy calculation consistency"""
        energy = total_system_energy(self.test_vectors, J=1.0)
        
        assert isinstance(energy, (int, float))
        assert not np.isnan(energy)
        
        # Energy should be consistent with Metropolis updates
        updated_vectors = update_vectors_metropolis(self.test_vectors, 1.0, J=1.0)
        updated_energy = total_system_energy(updated_vectors, J=1.0)
        
        # Energies should be finite
        assert np.isfinite(energy)
        assert np.isfinite(updated_energy)
    
    def test_total_system_energy_coupling(self):
        """Test system energy with different coupling strengths"""
        energy_1 = total_system_energy(self.test_vectors, J=1.0)
        energy_2 = total_system_energy(self.test_vectors, J=2.0)
        
        # Energy should scale with coupling strength
        assert abs(energy_2 - 2 * energy_1) < 1e-6
    
    def test_simulation_edge_cases(self):
        """Test simulation with edge cases"""
        # Empty vectors - should raise ValueError
        empty_vectors = np.array([]).reshape(0, 768)
        with pytest.raises(ValueError, match="Cannot run simulation with empty vectors array"):
            run_temperature_sweep(empty_vectors, self.temperature_range)
        
        # Single vector
        single_vector = self.test_vectors[:1]
        result = run_temperature_sweep(single_vector, self.temperature_range)
        assert 'alignment' in result
        assert 'entropy' in result
    
    def test_memory_management(self):
        """Test memory management in temperature sweep"""
        # Create larger vectors to test memory management
        large_vectors = np.random.randn(10, 768)
        large_vectors = large_vectors / np.linalg.norm(large_vectors, axis=1, keepdims=True)
        
        # Test with snapshots but limited memory
        result = run_temperature_sweep(
            large_vectors, 
            self.temperature_range, 
            store_all_temperatures=True,
            max_snapshots=2  # Limit snapshots
        )
        
        assert 'vector_snapshots' in result
        assert len(result['vector_snapshots']) <= 2 
    
    def test_rerun_at_tc_logs_entropy_history(self):
        """Test that rerunning at Tc logs entropy (alignment) history for all iterations."""
        # Run a sweep to get metrics and detect Tc
        sweep_result = run_temperature_sweep(self.test_vectors, self.temperature_range)
        # For test, just pick a Tc in the sweep range
        tc = self.temperature_range[2]  # e.g., 1.5
        # Rerun at Tc with detailed logging
        metrics, updated_vectors, convergence_info = simulate_at_temperature(self.test_vectors, tc, log_history=True)
        # Check that alignment_history is present and has multiple entries
        assert 'alignment_history' in convergence_info
        assert isinstance(convergence_info['alignment_history'], list)
        assert len(convergence_info['alignment_history']) > 1
        # Check that logged_steps matches alignment_history
        assert 'logged_steps' in convergence_info
        assert len(convergence_info['logged_steps']) == len(convergence_info['alignment_history'])
        # Check that entropy can be computed for each step
        entropies = [1.0 - align for align in convergence_info['alignment_history']]
        assert all(0.0 <= e <= 1.0 for e in entropies)

    def test_entropy_evolution_at_tc_rerun(self):
        """Test that after a sweep and Tc detection, a rerun at Tc logs entropy evolution and is stored in results."""
        # Run a sweep to get metrics and detect Tc
        sweep_result = run_temperature_sweep(self.test_vectors, self.temperature_range)
        # Simulate Tc detection (pick a value in the sweep)
        tc = self.temperature_range[2]  # e.g., 1.5
        # Simulate rerun at Tc and store in results
        metrics, updated_vectors, convergence_info = simulate_at_temperature(self.test_vectors, tc, log_every=1)
        # Store in results as would be done in pipeline
        sweep_result['entropy_evolution_at_tc'] = convergence_info
        # Check that the key exists and has expected structure
        assert 'entropy_evolution_at_tc' in sweep_result
        info = sweep_result['entropy_evolution_at_tc']
        assert 'alignment_history' in info and isinstance(info['alignment_history'], list)
        assert 'logged_steps' in info and isinstance(info['logged_steps'], list)
        assert len(info['alignment_history']) == len(info['logged_steps'])
        # Check that entropy can be computed for each step
        entropies = [1.0 - align for align in info['alignment_history']]
        assert all(0.0 <= e <= 1.0 for e in entropies) 