"""
Test suite for Phase 8: Export & I/O functionality

Tests all export functions including data export, results export, and UI helpers.
Following TDD principles - tests written before implementation.
"""

import pytest
import numpy as np
import tempfile
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import functions to be tested (will be implemented)
try:
    from export.io import save_json, save_csv, save_embeddings
    from export.results import export_results
    from export.ui_helpers import (
        export_csv_results, 
        export_vectors_at_tc, 
        export_charts, 
        export_config_file
    )
except ImportError:
    # Functions not yet implemented - this is expected in TDD
    pass


class TestExportIO:
    """Test basic I/O functions for data export"""
    
    def test_save_json_valid_data(self):
        """Test saving valid JSON data"""
        test_data = {
            'temperatures': [0.1, 0.5, 1.0],
            'alignment': [0.8, 0.6, 0.3],
            'metadata': {'concept': 'dog', 'encoder': 'LaBSE'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_json(test_data, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['temperatures'] == test_data['temperatures']
            assert loaded_data['alignment'] == test_data['alignment']
            assert loaded_data['metadata']['concept'] == 'dog'
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_json_invalid_filepath(self):
        """Test saving JSON with invalid filepath"""
        test_data = {'test': 'data'}
        
        # Use a path with invalid characters that will fail on Windows
        with pytest.raises(IOError):
            save_json(test_data, 'C:\\invalid\\path\\with\\<invalid>\\chars\\test.json')
    
    def test_save_json_non_serializable_data(self):
        """Test saving JSON with non-serializable data (should handle gracefully)"""
        test_data = {
            'numpy_array': np.array([1, 2, 3]),
            'datetime': datetime.now(),
            'regular_data': 'test'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should not raise exception, should handle non-serializable data
            save_json(test_data, temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_valid_data(self):
        """Test saving valid CSV data"""
        test_data = {
            'temperatures': [0.1, 0.5, 1.0],
            'alignment': [0.8, 0.6, 0.3],
            'entropy': [0.2, 0.4, 0.7]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_csv(test_data, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            df = pd.read_csv(temp_path)
            
            assert len(df) == 3
            assert list(df.columns) == ['temperatures', 'alignment', 'entropy']
            assert df['temperatures'].tolist() == test_data['temperatures']
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_empty_data(self):
        """Test saving empty CSV data"""
        test_data = {
            'temperatures': [],
            'alignment': [],
            'entropy': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_csv(test_data, temp_path)
            assert os.path.exists(temp_path)
            
            # Should create empty CSV with headers
            df = pd.read_csv(temp_path)
            assert len(df) == 0
            assert list(df.columns) == ['temperatures', 'alignment', 'entropy']
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_invalid_filepath(self):
        """Test saving CSV with invalid filepath"""
        test_data = {'test': [1, 2, 3]}
        
        # Use a path with invalid characters that will fail on Windows
        with pytest.raises(IOError):
            save_csv(test_data, 'C:\\invalid\\path\\with\\<invalid>\\chars\\test.csv')
    
    def test_save_embeddings_numpy_format(self):
        """Test saving embeddings in NumPy format"""
        vectors = np.random.randn(5, 768)
        filename = "test_embeddings"
        filetype = "meta"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = save_embeddings(vectors, filename, filetype, temp_dir)
            
            # Verify files were created
            expected_npy = os.path.join(temp_dir, f"{filename}.npy")
            expected_metadata = os.path.join(temp_dir, f"{filename}_metadata.json")
            
            assert os.path.exists(expected_npy)
            assert os.path.exists(expected_metadata)
            assert filepath == expected_npy
            
            # Verify data integrity
            loaded_vectors = np.load(expected_npy)
            assert np.array_equal(vectors, loaded_vectors)
            
            # Verify metadata
            with open(expected_metadata, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['shape'] == list(vectors.shape)
            assert metadata['filetype'] == filetype
    
    def test_save_embeddings_invalid_vectors(self):
        """Test saving invalid embeddings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                save_embeddings(None, "test", "meta", temp_dir)
    
    def test_save_embeddings_invalid_output_dir(self):
        """Test saving embeddings with invalid output directory"""
        vectors = np.random.randn(5, 768)
        
        # Use a path with invalid characters that will fail on Windows
        with pytest.raises(IOError):
            save_embeddings(vectors, "test", "meta", "C:\\invalid\\path\\with\\<invalid>\\chars")


class TestExportResults:
    """Test comprehensive results export functionality"""
    
    def test_export_results_basic(self):
        """Test basic results export without comparison metrics"""
        metrics = {
            'temperatures': np.array([0.1, 0.5, 1.0]),
            'alignment': np.array([0.8, 0.6, 0.3]),
            'entropy': np.array([0.2, 0.4, 0.7])
        }
        tc = 0.5
        meta_result = {
            'meta_vector': np.random.randn(768),
            'method': 'centroid'
        }
        concept = 'dog'
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_results(metrics, tc, meta_result, concept, temp_dir)
            
            # Verify files were created
            expected_csv = os.path.join(temp_dir, f"{concept}_metrics.csv")
            expected_json = os.path.join(temp_dir, f"{concept}_summary.json")
            expected_meta = os.path.join(temp_dir, f"{concept}_meta.npy")
            
            assert os.path.exists(expected_csv)
            assert os.path.exists(expected_json)
            assert os.path.exists(expected_meta)
            
            # Verify summary JSON content
            with open(expected_json, 'r') as f:
                summary = json.load(f)
            
            assert summary['concept'] == concept
            assert summary['critical_temperature'] == tc
            assert summary['meta_vector_method'] == 'centroid'
    
    def test_export_results_with_comparison(self):
        """Test results export with anchor comparison metrics"""
        metrics = {
            'temperatures': np.array([0.1, 0.5, 1.0]),
            'alignment': np.array([0.8, 0.6, 0.3])
        }
        tc = 0.5
        meta_result = {
            'meta_vector': np.random.randn(768),
            'method': 'centroid'
        }
        concept = 'dog'
        comparison_metrics = {
            'procrustes_distance': 0.123,
            'cka_similarity': 0.789,
            'emd_distance': 0.456,
            'kl_divergence': 0.321,
            'cosine_similarity': 0.654
        }
        experiment_config = {
            'anchor_language': 'en',
            'include_anchor': False,
            'dynamics_languages': ['es', 'fr', 'de']
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_results(metrics, tc, meta_result, concept, temp_dir, 
                          comparison_metrics, experiment_config)
            
            # Verify comparison report was created
            expected_comparison = os.path.join(temp_dir, f"{concept}_anchor_comparison.json")
            assert os.path.exists(expected_comparison)
            
            # Verify comparison report content
            with open(expected_comparison, 'r') as f:
                comparison_report = json.load(f)
            
            assert comparison_report['concept'] == concept
            assert comparison_report['critical_temperature'] == tc
            assert comparison_report['anchor_comparison_metrics'] == comparison_metrics
            assert comparison_report['experiment_config'] == experiment_config
    
    def test_export_results_invalid_output_dir(self):
        """Test results export with invalid output directory"""
        metrics = {'temperatures': np.array([0.1, 0.5])}
        tc = 0.5
        meta_result = {'meta_vector': np.random.randn(768), 'method': 'centroid'}
        
        # Use a path with invalid characters that will fail on Windows
        with pytest.raises(IOError):
            export_results(metrics, tc, meta_result, 'dog', 'C:\\invalid\\path\\with\\<invalid>\\chars')


class TestExportUIHelpers:
    """Test UI helper functions for Streamlit interface"""
    
    def test_export_csv_results(self):
        """Test CSV export for UI"""
        simulation_results = {
            'metrics': {
                'temperatures': np.array([0.1, 0.5, 1.0]),
                'alignment': np.array([0.8, 0.6, 0.3])
            }
        }
        analysis_results = {
            'anchor_comparison': {
                'cka_similarity': 0.789,
                'procrustes_distance': 0.123
            }
        }
        
        filepath = export_csv_results(simulation_results, analysis_results)
        
        # Verify temporary file was created
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify CSV content
        df = pd.read_csv(filepath)
        assert len(df) == 3  # 3 temperature points
        assert 'temperatures' in df.columns
        assert 'alignment' in df.columns
        
        # Clean up
        os.unlink(filepath)
    
    def test_export_vectors_at_tc_with_snapshots(self):
        """Test vector export at critical temperature with snapshots"""
        simulation_results = {
            'critical_temperature': 0.5,
            'vector_snapshots': {
                0.5: np.random.randn(5, 768)
            }
        }
        
        filepath = export_vectors_at_tc(simulation_results)
        
        # Verify temporary file was created
        assert os.path.exists(filepath)
        assert filepath.endswith('.npy')
        
        # Verify data integrity
        loaded_vectors = np.load(filepath)
        assert loaded_vectors.shape == (5, 768)
        
        # Clean up
        os.unlink(filepath)
    
    def test_export_vectors_at_tc_no_snapshots(self):
        """Test vector export when no snapshots available"""
        simulation_results = {
            'critical_temperature': 0.5
            # No vector_snapshots
        }
        
        with pytest.raises(ValueError):
            export_vectors_at_tc(simulation_results)
    
    def test_export_charts(self):
        """Test chart export functionality"""
        simulation_results = {
            'metrics': {
                'temperatures': np.array([0.1, 0.5, 1.0]),
                'alignment': np.array([0.8, 0.6, 0.3])
            }
        }
        analysis_results = {
            'anchor_comparison': {
                'cka_similarity': 0.789
            }
        }

        filepath = export_charts(simulation_results, analysis_results)

        # Verify temporary file was created
        assert os.path.exists(filepath)
        if filepath.endswith('.png'):
            # Chart export succeeded
            assert os.path.getsize(filepath) > 0
        elif filepath.endswith('.txt'):
            # Fallback: check for kaleido missing message
            with open(filepath, 'r') as f:
                content = f.read()
            assert 'kaleido' in content.lower()
        else:
            assert False, f"Unexpected file type: {filepath}"

        # Clean up
        os.unlink(filepath)
    
    def test_export_config_file(self):
        """Test configuration file export"""
        filepath = export_config_file()
        
        # Verify temporary file was created
        assert os.path.exists(filepath)
        assert filepath.endswith('.yaml')
        
        # Verify YAML content
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert 'temperature_range' in content
        assert 'default_encoder' in content
        
        # Clean up
        os.unlink(filepath)


class TestExportEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_save_json_empty_data(self):
        """Test saving empty JSON data"""
        test_data = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_json(test_data, temp_path)
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == {}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_csv_mismatched_lengths(self):
        """Test saving CSV with mismatched data lengths"""
        test_data = {
            'temperatures': [0.1, 0.5, 1.0],
            'alignment': [0.8, 0.6]  # Different length
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should raise IOError for mismatched lengths
            with pytest.raises(IOError):
                save_csv(test_data, temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_embeddings_zero_vectors(self):
        """Test saving zero-dimensional embeddings"""
        vectors = np.array([])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                save_embeddings(vectors, "test", "meta", temp_dir)
    
    def test_export_results_missing_meta_vector(self):
        """Test results export with missing meta vector"""
        metrics = {'temperatures': np.array([0.1, 0.5])}
        tc = 0.5
        meta_result = {}  # Missing meta_vector
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(KeyError):
                export_results(metrics, tc, meta_result, 'dog', temp_dir)


class TestExportIntegration:
    """Test integration with existing functionality"""
    
    def test_export_with_post_analysis_results(self):
        """Test export integration with post-analysis results"""
        # Mock post-analysis results structure
        analysis_results = {
            'critical_temperature': 0.5,
            'anchor_comparison': {
                'procrustes_distance': 0.123,
                'cka_similarity': 0.789,
                'emd_distance': 0.456,
                'kl_divergence': 0.321,
                'cosine_similarity': 0.654
            },
            'power_law_analysis': {
                'exponent': 2.1,
                'r_squared': 0.85,
                'n_clusters': 5
            }
        }
        
        # Test that export functions can handle this structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should work without errors
            pass  # Placeholder for actual integration test
    
    def test_export_with_simulation_results(self):
        """Test export integration with simulation results"""
        # Mock simulation results structure
        simulation_results = {
            'metrics': {
                'temperatures': np.array([0.1, 0.5, 1.0, 1.5, 2.0]),
                'alignment': np.array([0.9, 0.7, 0.4, 0.2, 0.1]),
                'entropy': np.array([0.1, 0.3, 0.6, 0.8, 0.9]),
                'energy': np.array([-0.9, -0.7, -0.4, -0.2, -0.1]),
                'correlation_length': np.array([0.5, 1.2, 2.1, 1.8, 1.3])
            },
            'critical_temperature': 1.0,
            'vector_snapshots': {
                1.0: np.random.randn(10, 768)
            }
        }
        
        # Test that export functions can handle this structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should work without errors
            pass  # Placeholder for actual integration test 