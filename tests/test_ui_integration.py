"""
Integration tests for UI components (Phase 9)
Tests the complete UI pipeline without complex mocking
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI components
from ui.charts import (
    plot_entropy_vs_temperature,
    plot_alignment_vs_temperature,
    plot_energy_vs_temperature,
    plot_correlation_length_vs_temperature,
    plot_correlation_decay,
    plot_full_umap_projection
)

from ui.components import (
    render_anchor_config,
    render_experiment_description,
    render_metrics_summary,
    render_critical_temperature_display,
    render_anchor_comparison_summary,
    render_power_law_summary,
    render_export_buttons,
    render_error_message,
    render_success_message,
    render_warning_message
)


class TestUIChartGeneration:
    """Test chart generation functions"""
    
    def test_plot_entropy_vs_temperature_with_data(self):
        """Test entropy plot with valid data"""
        # Create sample data
        results = {
            'metrics': {
                'temperatures': np.array([0.5, 1.0, 1.5, 2.0]),
                'entropy': np.array([0.1, 0.3, 0.5, 0.7])
            },
            'critical_temperature': 1.2
        }
        
        fig = plot_entropy_vs_temperature(results)
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_plot_alignment_vs_temperature_with_data(self):
        """Test alignment plot with valid data"""
        results = {
            'metrics': {
                'temperatures': np.array([0.5, 1.0, 1.5, 2.0]),
                'alignment': np.array([0.9, 0.7, 0.5, 0.3])
            }
        }
        
        fig = plot_alignment_vs_temperature(results)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_energy_vs_temperature_with_data(self):
        """Test energy plot with valid data"""
        results = {
            'metrics': {
                'temperatures': np.array([0.5, 1.0, 1.5, 2.0]),
                'energy': np.array([-2.0, -1.5, -1.0, -0.5])
            }
        }
        
        fig = plot_energy_vs_temperature(results)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_correlation_length_vs_temperature_with_data(self):
        """Test correlation length plot with valid data"""
        results = {
            'metrics': {
                'temperatures': np.array([0.5, 1.0, 1.5, 2.0]),
                'correlation_length': np.array([0.1, 0.3, 0.2, 0.1])
            }
        }
        
        fig = plot_correlation_length_vs_temperature(results)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_correlation_decay_with_data(self):
        """Test correlation decay plot with valid data"""
        analysis_results = {
            'correlation_analysis': {
                'distances': np.array([0, 1, 2, 3]),
                'correlations': np.array([1.0, 0.8, 0.6, 0.4])
            }
        }
        
        fig = plot_correlation_decay(analysis_results)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_plot_full_umap_projection_with_data(self):
        """Test UMAP projection plot with valid data"""
        analysis_results = {
            'umap_projection': {
                'embeddings_2d': np.random.rand(10, 2),
                'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                'cluster_labels': np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            }
        }
        
        # Test without anchor language highlighting
        fig = plot_full_umap_projection(analysis_results, analysis_results)
        assert fig is not None
        assert hasattr(fig, 'data')
        
        # Test with anchor language highlighting
        fig = plot_full_umap_projection(analysis_results, analysis_results, 
                                      anchor_language='en', include_anchor=True)
        assert fig is not None
        assert hasattr(fig, 'data')


class TestUIComponentFunctions:
    """Test UI component functions"""
    
    def test_render_anchor_config_returns_tuple(self):
        """Test that render_anchor_config returns expected tuple"""
        with patch('streamlit.sidebar'), patch('streamlit.selectbox') as mock_selectbox, patch('streamlit.checkbox') as mock_checkbox, patch('streamlit.info'):
            # Mock return values
            mock_selectbox.return_value = "en"
            mock_checkbox.return_value = False
            
            anchor_language, include_anchor = render_anchor_config()
            assert isinstance(anchor_language, str)
            assert isinstance(include_anchor, bool)
    
    def test_render_experiment_description_no_error(self):
        """Test that render_experiment_description doesn't raise errors"""
        with patch('streamlit.info'):
            try:
                render_experiment_description("en", False, ["es", "fr", "de"])
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_metrics_summary_no_error(self):
        """Test that render_metrics_summary doesn't raise errors"""
        with patch('streamlit.subheader'), patch('streamlit.columns'), patch('streamlit.metric'):
            try:
                metrics = {'alignment': 0.8, 'entropy': 0.5}
                render_metrics_summary(metrics)
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_critical_temperature_display_no_error(self):
        """Test that render_critical_temperature_display doesn't raise errors"""
        with patch('streamlit.subheader'), patch('streamlit.columns'), patch('streamlit.metric'):
            try:
                render_critical_temperature_display(1.2, "Binder Cumulant")
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_anchor_comparison_summary_no_error(self):
        """Test that render_anchor_comparison_summary doesn't raise errors"""
        with patch('streamlit.subheader'), patch('streamlit.columns'), patch('streamlit.metric'):
            try:
                comparison_metrics = {
                    'procrustes_distance': 0.1,
                    'cka_similarity': 0.8,
                    'emd_distance': 0.2,
                    'cosine_similarity': 0.9
                }
                render_anchor_comparison_summary(comparison_metrics)
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_power_law_summary_no_error(self):
        """Test that render_power_law_summary doesn't raise errors"""
        with patch('streamlit.subheader'), patch('streamlit.columns'), patch('streamlit.metric'):
            try:
                power_law_analysis = {
                    'exponent': 2.1,
                    'r_squared': 0.95,
                    'n_clusters': 3
                }
                render_power_law_summary(power_law_analysis)
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_export_buttons_no_error(self):
        """Test that render_export_buttons doesn't raise errors"""
        with patch('streamlit.subheader'), patch('streamlit.columns'), patch('streamlit.button'):
            try:
                simulation_results = {'metrics': {'temperatures': [0.5, 1.0]}}
                analysis_results = {'anchor_comparison': {'cka_similarity': 0.8}}
                render_export_buttons(simulation_results, analysis_results)
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_error_message_no_error(self):
        """Test that render_error_message doesn't raise errors"""
        with patch('streamlit.error'):
            try:
                render_error_message(Exception("Test error"), "test context")
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_success_message_no_error(self):
        """Test that render_success_message doesn't raise errors"""
        with patch('streamlit.success'):
            try:
                render_success_message("Test success")
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")
    
    def test_render_warning_message_no_error(self):
        """Test that render_warning_message doesn't raise errors"""
        with patch('streamlit.warning'):
            try:
                render_warning_message("Test warning")
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")


class TestUITabFunctions:
    """Test UI tab functions"""
    
    def test_simulation_tab_imports(self):
        """Test that simulation tab can be imported"""
        try:
            from ui.tabs.simulation import render_simulation_tab
            assert callable(render_simulation_tab)
        except ImportError as e:
            pytest.fail(f"Failed to import simulation tab: {e}")
    
    def test_metrics_export_tab_imports(self):
        """Test that metrics export tab can be imported"""
        try:
            from ui.tabs.metrics_export import render_metrics_export_tab
            assert callable(render_metrics_export_tab)
        except ImportError as e:
            pytest.fail(f"Failed to import metrics export tab: {e}")
    
    def test_anchor_comparison_tab_imports(self):
        """Test that anchor comparison tab can be imported"""
        try:
            from ui.tabs.anchor_comparison import render_anchor_comparison_tab
            assert callable(render_anchor_comparison_tab)
        except ImportError as e:
            pytest.fail(f"Failed to import anchor comparison tab: {e}")


class TestMainAppIntegration:
    """Test main app integration"""
    
    def test_main_app_imports(self):
        """Test that main app can be imported"""
        try:
            from app import main, render_overview_tab
            assert callable(main)
            assert callable(render_overview_tab)
        except ImportError as e:
            pytest.fail(f"Failed to import main app: {e}")
    
    def test_main_app_functions_no_error(self):
        """Test that main app functions don't raise errors"""
        with patch('streamlit.set_page_config'), patch('streamlit.title'), patch('streamlit.markdown'), \
             patch('streamlit.sidebar'), patch('streamlit.tabs'), patch('streamlit.columns'):
            try:
                from app import render_overview_tab
                render_overview_tab("dog", "LaBSE", [0.5, 1.0, 1.5], "en", False)
                assert True  # No error raised
            except Exception as e:
                pytest.fail(f"Function raised unexpected error: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 