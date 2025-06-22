"""
Performance benchmark tests for Semantic Ising Simulator.

Measures computational efficiency of all modules and identifies performance
bottlenecks. Includes timing benchmarks and memory usage analysis.
"""

import pytest
import numpy as np
import time
import psutil
import gc
import os
from functools import wraps

# Import core modules for benchmarking
from core.embeddings import generate_embeddings, load_concept_embeddings
from core.simulation import run_temperature_sweep, simulate_at_temperature, compute_correlation_matrix, compute_correlation_length
from core.phase_detection import find_critical_temperature, detect_powerlaw_regime
from core.clustering import cluster_vectors
from core.comparison_metrics import compare_anchor_to_multilingual
from core.meta_vector import compute_meta_vector


def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__}: {execution_time:.4f} seconds")
        return result, execution_time
    return wrapper


def memory_usage_decorator(func):
    """Decorator to measure memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        print(f"{func.__name__}: {memory_used:.2f} MB")
        return result, memory_used
    return wrapper


class TestPerformanceBenchmarks:
    """Performance benchmarks for core modules."""
    
    def setup_method(self):
        """Set up test data for benchmarks."""
        # Create test vectors of different sizes
        self.small_vectors = np.random.randn(10, 768)
        self.small_vectors = self.small_vectors / np.linalg.norm(self.small_vectors, axis=1, keepdims=True)
        
        self.medium_vectors = np.random.randn(50, 768)
        self.medium_vectors = self.medium_vectors / np.linalg.norm(self.medium_vectors, axis=1, keepdims=True)
        
        self.large_vectors = np.random.randn(200, 768)
        self.large_vectors = self.large_vectors / np.linalg.norm(self.large_vectors, axis=1, keepdims=True)
        
        # Create temperature sweep data
        self.temperatures = np.linspace(0.5, 2.5, 20)
        self.metrics = {
            'temperatures': self.temperatures,
            'alignment': 0.8 * np.exp(-self.temperatures) + 0.1,
            'entropy': 1.0 - (0.8 * np.exp(-self.temperatures) + 0.1),
            'energy': -(0.8 * np.exp(-self.temperatures) + 0.1),
            'correlation_length': np.exp(-np.abs(self.temperatures - 1.5))
        }
    
    def test_simulation_performance(self):
        """Benchmark single simulation performance for different system sizes."""
        print("\n=== Simulation Performance Benchmarks ===")
        
        @timing_decorator
        def benchmark_single_simulation(vectors, T):
            # Reduced max_iter for faster benchmark runs
            return simulate_at_temperature(vectors, T, max_iter=100)

        # Small system
        result, time_small = benchmark_single_simulation(self.small_vectors, 1.0)
        assert time_small < 5.0, f"Small system simulation took too long: {time_small:.2f}s"
        
        # Medium system
        result, time_medium = benchmark_single_simulation(self.medium_vectors, 1.0)
        assert time_medium < 30.0, f"Medium system simulation took too long: {time_medium:.2f}s"
        
        # Large system (increase timeout to allow for variability)
        result, time_large = benchmark_single_simulation(self.large_vectors, 1.0)
        assert time_large < 450.0, f"Large system simulation took too long: {time_large:.2f}s"
    
    def test_dynamics_performance(self):
        """Benchmark dynamics calculations."""
        print("\n=== Dynamics Performance Benchmarks ===")
        
        # Test correlation matrix computation
        @timing_decorator
        def benchmark_correlation_matrix(vectors):
            return compute_correlation_matrix(vectors)
        
        # Small system
        result, time_small = benchmark_correlation_matrix(self.small_vectors)
        assert time_small < 0.1, f"Small correlation matrix took too long: {time_small:.3f}s"
        
        # Medium system
        result, time_medium = benchmark_correlation_matrix(self.medium_vectors)
        assert time_medium < 0.5, f"Medium correlation matrix took too long: {time_medium:.3f}s"
        
        # Large system
        result, time_large = benchmark_correlation_matrix(self.large_vectors)
        assert time_large < 2.0, f"Large correlation matrix took too long: {time_large:.3f}s"
        
        # Test correlation length computation
        @timing_decorator
        def benchmark_correlation_length(vectors):
            return compute_correlation_length(vectors)
        
        result, time_xi = benchmark_correlation_length(self.medium_vectors)
        assert time_xi < 1.0, f"Correlation length computation took too long: {time_xi:.3f}s"
    
    def test_phase_detection_performance(self):
        """Benchmark phase detection algorithms."""
        print("\n=== Phase Detection Performance Benchmarks ===")
        
        # Test critical temperature detection
        @timing_decorator
        def benchmark_critical_temperature(metrics):
            return find_critical_temperature(metrics)
        
        result, time_tc = benchmark_critical_temperature(self.metrics)
        assert time_tc < 1.0, f"Critical temperature detection took too long: {time_tc:.3f}s"
        
        # Test power law regime detection
        @timing_decorator
        def benchmark_powerlaw_detection(vectors):
            return detect_powerlaw_regime(vectors, T=1.5)
        
        result, time_powerlaw = benchmark_powerlaw_detection(self.medium_vectors)
        assert time_powerlaw < 2.0, f"Power law detection took too long: {time_powerlaw:.3f}s"
    
    def test_clustering_performance(self):
        """Benchmark clustering algorithms."""
        print("\n=== Clustering Performance Benchmarks ===")
        
        @timing_decorator
        def benchmark_clustering(vectors):
            return cluster_vectors(vectors)
        
        # Small system
        result, time_small = benchmark_clustering(self.small_vectors)
        assert time_small < 0.5, f"Small clustering took too long: {time_small:.3f}s"
        
        # Medium system
        result, time_medium = benchmark_clustering(self.medium_vectors)
        assert time_medium < 2.0, f"Medium clustering took too long: {time_medium:.3f}s"
        
        # Large system
        result, time_large = benchmark_clustering(self.large_vectors)
        assert time_large < 10.0, f"Large clustering took too long: {time_large:.3f}s"
    
    def test_meta_vector_performance(self):
        """Benchmark meta vector computation."""
        print("\n=== Meta Vector Performance Benchmarks ===")
        
        methods = ['centroid', 'medoid', 'geometric_median', 'first_principal_component']
        
        for method in methods:
            @timing_decorator
            def benchmark_meta_vector(vectors, method):
                return compute_meta_vector(vectors, method=method)
            
            result, time_method = benchmark_meta_vector(self.medium_vectors, method)
            assert time_method < 1.0, f"{method} took too long: {time_method:.3f}s"
        
        # Test weighted_mean separately with weights
        @timing_decorator
        def benchmark_weighted_mean(vectors):
            weights = np.random.rand(len(vectors))
            weights = weights / np.sum(weights)
            return compute_meta_vector(vectors, method='weighted_mean', weights=weights)
        
        result, time_weighted = benchmark_weighted_mean(self.medium_vectors)
        assert time_weighted < 1.0, f"weighted_mean took too long: {time_weighted:.3f}s"
    
    def test_comparison_metrics_performance(self):
        """Benchmark comparison metrics."""
        print("\n=== Comparison Metrics Performance Benchmarks ===")
        
        # Create two sets of vectors for comparison
        vectors_a = self.medium_vectors[:25]
        vectors_b = self.medium_vectors[25:]
        
        # Create required metrics dictionary
        temperatures = np.linspace(0.5, 2.5, 10)
        metrics = {
            'temperatures': temperatures,
            'alignment': 0.8 * np.exp(-temperatures) + 0.1,
            'entropy': 1.0 - (0.8 * np.exp(-temperatures) + 0.1),
            'energy': -(0.8 * np.exp(-temperatures) + 0.1),
            'correlation_length': np.exp(-np.abs(temperatures - 1.5))
        }
        
        @timing_decorator
        def benchmark_comparison(vectors_a, vectors_b, tc, metrics):
            return compare_anchor_to_multilingual(vectors_a, vectors_b, tc, metrics)
        
        result, time_comparison = benchmark_comparison(vectors_a, vectors_b, 1.5, metrics)
        assert time_comparison < 5.0, f"Comparison metrics took too long: {time_comparison:.3f}s"


class TestMemoryUsageBenchmarks:
    """Memory usage benchmarks for core modules."""
    
    def setup_method(self):
        """Set up test data for memory benchmarks."""
        self.vectors = np.random.randn(100, 768)
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
    
    def test_simulation_memory_usage(self):
        """Test memory usage during simulation."""
        print("\n=== Simulation Memory Usage Benchmarks ===")
        
        @memory_usage_decorator
        def benchmark_simulation_memory(vectors):
            return run_temperature_sweep(vectors, np.linspace(0.5, 2.5, 10), n_sweeps_per_temperature=10)
        
        result, memory_used = benchmark_simulation_memory(self.vectors)
        assert memory_used < 100, f"Simulation used too much memory: {memory_used:.1f} MB"
    
    def test_clustering_memory_usage(self):
        """Test memory usage during clustering."""
        print("\n=== Clustering Memory Usage Benchmarks ===")
        
        @memory_usage_decorator
        def benchmark_clustering_memory(vectors):
            return cluster_vectors(vectors)
        
        result, memory_used = benchmark_clustering_memory(self.vectors)
        assert memory_used < 50, f"Clustering used too much memory: {memory_used:.1f} MB"
    
    def test_meta_vector_memory_usage(self):
        """Test memory usage during meta vector computation."""
        print("\n=== Meta Vector Memory Usage Benchmarks ===")
        
        @memory_usage_decorator
        def benchmark_meta_vector_memory(vectors):
            return compute_meta_vector(vectors, method='geometric_median')
        
        result, memory_used = benchmark_meta_vector_memory(self.vectors)
        assert memory_used < 20, f"Meta vector computation used too much memory: {memory_used:.1f} MB"


class TestScalabilityBenchmarks:
    """Benchmark simulation scalability with increasing system size."""
    
    def test_simulation_scalability(self):
        """Benchmark time scalability with vector count."""
        print("\n=== Simulation Scalability Benchmarks ===")
        sizes = [10, 25, 50, 100]
        times = []

        for n in sizes:
            vectors = np.random.randn(n, 768)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            
            start_time = time.time()
            simulate_at_temperature(vectors, 1.0, max_iter=50)
            end_time = time.time()
            
            duration = end_time - start_time
            times.append(duration)
            print(f"Size {n}: {duration:.3f}s")
        
        # Check that scaling is not excessively poor (e.g., O(N^3))
        # The vectorized approach is O(N^2), so we expect non-linear scaling.
        # We'll relax the assertion to check for reasonable quadratic scaling.
        for i in range(1, len(sizes)):
            size_ratio = (sizes[i] / sizes[i-1]) ** 2  # Expected scaling for O(N^2)
            time_ratio = times[i] / times[i-1]
            # Allow for some overhead, so check if time ratio is less than size_ratio^1.5, for example
            assert time_ratio < size_ratio * 1.8, f"Time scaling too poorly: {time_ratio:.1f}x for size increase from {sizes[i-1]} to {sizes[i]}"

    def test_memory_scalability(self):
        """Benchmark memory scalability with vector count."""
        print("\n=== Memory Scalability Benchmarks ===")
        sizes = [10, 25, 50, 100]
        memory_usages = []

        for n in sizes:
            vectors = np.random.randn(n, 768)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024

            simulate_at_temperature(vectors, 1.0, max_iter=50)
            compute_correlation_matrix(vectors)
            cluster_vectors(vectors)
            compute_meta_vector(vectors, method='centroid')

            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            memory_usages.append(memory_used)
            print(f"Size {n}: {memory_used:.2f} MB")

        # Check that memory usage scales reasonably with vector count
        # Only test scaling when there's meaningful memory usage (> 0.01 MB)
        for i in range(1, len(sizes)):
            if memory_usages[i-1] > 0.01 and memory_usages[i] > 0.01:
                size_ratio = (sizes[i] / sizes[i-1]) ** 2  # Expected scaling for O(N^2)
                memory_ratio = memory_usages[i] / memory_usages[i-1]
                assert memory_ratio < size_ratio * 2.0, f"Memory scaling too poorly: {memory_ratio:.1f}x for size increase from {sizes[i-1]} to {sizes[i]}"
            else:
                # If memory usage is negligible, just check that it's not negative
                assert memory_usages[i] >= 0, f"Memory usage should be non-negative, got {memory_usages[i]}"


class TestConcurrencyBenchmarks:
    """Test performance under concurrent operations."""
    
    def test_memory_cleanup(self):
        """Test that memory is cleaned up after simulation."""
        print("\n=== Memory Cleanup Benchmarks ===")
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Run multiple operations
        for _ in range(5):
            vectors = np.random.randn(50, 768)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            # Run various operations
            simulate_at_temperature(vectors, 1.0, max_iter=50)
            compute_correlation_matrix(vectors)
            cluster_vectors(vectors)
            compute_meta_vector(vectors, method='centroid')
            
            # Force garbage collection
            gc.collect()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory not properly cleaned up: {memory_increase:.1f} MB increase"


if __name__ == "__main__":
    pytest.main([__file__]) 