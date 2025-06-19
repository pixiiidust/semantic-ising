#!/usr/bin/env python3
"""
Semantic Ising Simulator - Streamlit Dashboard

This dashboard provides an interactive interface for running semantic Ising simulations
and analyzing multilingual embedding convergence under Ising dynamics.
"""

import os, types, sys
# 1) Tell Streamlit to use the "watchdog" watcher (no introspection of __path__)
os.environ.setdefault("STREAMLIT_WATCHER_TYPE", "watchdog")
# 2) Set environment variable to prevent PyTorch introspection issues
os.environ.setdefault("PYTORCH_JIT", "0")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PyTorch/Streamlit compatibility fix
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch")

# Try to import PyTorch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False
except Exception as e:
    print(f"Warning: PyTorch import error: {e}")
    TORCH_AVAILABLE = False

# Import core modules
from core.embeddings import generate_embeddings
from core.anchor_config import configure_anchor_experiment, get_experiment_description
from core.simulation import run_temperature_sweep, simulate_at_temperature
from core.phase_detection import find_critical_temperature
from core.post_analysis import analyze_simulation_results
from core.temperature_estimation import estimate_practical_range
from ui.charts import plot_entropy_vs_temperature, plot_full_umap_projection, plot_correlation_decay, plot_correlation_length_vs_temperature
from ui.components import render_anchor_config, render_experiment_description, render_simulation_progress, render_metrics_summary, render_critical_temperature_display, render_anchor_comparison_summary, render_export_buttons, render_error_message, render_success_message, render_warning_message, render_concept_selection
from ui.tabs.simulation import render_simulation_tab
from ui.tabs.anchor_comparison import render_anchor_comparison_tab
from ui.tabs.overview import render_overview_tab

logger = logging.getLogger(__name__)


def load_config_defaults():
    """Load default configuration values from YAML file"""
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'defaults.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract temperature-related defaults
        temp_range = config.get('temperature_range', [0.1, 3.0])
        temp_steps = config.get('temperature_steps', 50)
        
        return {
            'temp_min': temp_range[0],
            'temp_max': temp_range[1], 
            'temp_steps': temp_steps
        }
    except Exception as e:
        logger.warning(f"Failed to load config defaults: {e}")
        # Fallback defaults
        return {
            'temp_min': 0.1,
            'temp_max': 3.0,
            'temp_steps': 50
        }


def main():
    """Main Streamlit application"""
    try:
        # Configure page
        st.set_page_config(
            page_title="Semantic Ising Simulator",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load config defaults
        config_defaults = load_config_defaults()
        
        # Title and description
        st.title("🧠 Semantic Ising Simulator")
        st.markdown("""
        **Multilingual semantic convergence analysis using Ising dynamics**
    
        * This simulator tests whether semantically identical words across languages 
        converge in embedding space under Ising dynamics. 
        * Results potentially reveal universal semantic structures approaching critical temperature.
        """)
        
        # Test mode toggle (for debugging)
        test_mode = st.sidebar.checkbox("Test Mode", value=False, help="Run in test mode with mock data")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Dynamic concept selection
            concept, concept_info = render_concept_selection()
            
            # Only show other configuration if a concept is selected
            if concept is None:
                st.stop()
            
            # Encoder selection
            encoder = st.selectbox(
                "Embedding Model",
                ["LaBSE", "mBERT (coming soon)", "XLM-R (coming soon)"],
                help="Select the multilingual embedding model"
            )
            
            # Anchor configuration
            anchor_language, include_anchor = render_anchor_config()
            
            # Temperature range configuration
            st.subheader("🌡️ Temperature Range")
            
            # Show config source info
            st.info(f"📋 Defaults from config: {config_defaults['temp_min']:.1f}-{config_defaults['temp_max']:.1f}, {config_defaults['temp_steps']} steps")
            
            # Auto-estimate checkbox
            use_auto_estimate = st.checkbox(
                "Auto-estimate temperature range (recommended)",
                value=True,
                help="Automatically estimate optimal temperature range based on vector properties"
            )
            
            if use_auto_estimate:
                # Try to estimate range immediately
                try:
                    # Generate embeddings for estimation
                    filename = concept_info.get('filename') if concept_info else None
                    embeddings, languages = generate_embeddings(concept, encoder, filename)
                    
                    # Configure anchor experiment for estimation
                    dynamics_languages, comparison_languages = configure_anchor_experiment(
                        languages, anchor_language, include_anchor
                    )
                    
                    # Extract embeddings for dynamics
                    dynamics_indices = [languages.index(lang) for lang in dynamics_languages]
                    dynamics_embeddings = embeddings[dynamics_indices]
                    
                    # Estimate temperature range
                    estimated_tmin, estimated_tmax = estimate_practical_range(
                        dynamics_embeddings, 
                        config_max_temperature=config_defaults['temp_max']
                    )
                    
                    # Store in session state
                    st.session_state.estimated_range = (estimated_tmin, estimated_tmax)
                    
                    # Show estimated range info
                    st.info(f"""
                    **Estimated Range:**
                    - Min: {estimated_tmin:.3f}
                    - Max: {estimated_tmax:.3f}
                    """)
                    
                    # Use estimated range for simulation
                    tmin = estimated_tmin
                    tmax = estimated_tmax
                    
                except Exception as e:
                    st.warning(f"⚠️ Temperature estimation failed: {e}")
                    st.info("Using default range. You can uncheck auto-estimate to set manually.")
                    # Fallback to config defaults
                    tmin = config_defaults['temp_min']
                    tmax = config_defaults['temp_max']
            else:
                # Manual temperature controls
                col1, col2 = st.columns(2)
                with col1:
                    tmin = st.number_input(
                        "Min Temperature",
                        min_value=0.01,
                        max_value=10.0,
                        value=config_defaults['temp_min'],  # Use config default
                        step=0.1,
                        format="%.2f"
                    )
                with col2:
                    tmax = st.number_input(
                        "Max Temperature", 
                        min_value=0.01,
                        max_value=20.0,
                        value=config_defaults['temp_max'],  # Use config default
                        step=0.1,
                        format="%.2f"
                    )
            
            # Number of temperature steps
            n_steps = st.slider(
                "Number of Steps",
                min_value=5,
                max_value=200,
                value=config_defaults['temp_steps'],  # Use config default
                help="Number of temperature points to simulate: More steps = more accurate results, but slower simulation"
            )
            
            # Create temperature range
            T_range = np.linspace(tmin, tmax, n_steps).tolist()
            
            # Run simulation button
            if st.button("🚀 Run Simulation", help="Start the semantic Ising simulation"):
                if test_mode:
                    # Create mock data for testing
                    create_mock_simulation_results(concept, encoder, T_range, anchor_language, include_anchor, concept_info)
                else:
                    run_simulation_workflow(concept, encoder, T_range, anchor_language, include_anchor, concept_info)
        
        # Main content area
        tab1, tab2, tab3 = st.tabs([
            "📋 Overview", 
            "⚙️ Simulation Results", 
            "🔗 Anchor Language Comparison"
        ])
        
        with tab1:
            render_overview_tab(concept, encoder, T_range, anchor_language, include_anchor)
        
        with tab2:
            render_simulation_tab(concept, encoder, T_range, anchor_language, include_anchor)
        
        with tab3:
            # Check if we have comparison data available
            if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                comparison_metrics = st.session_state.analysis_results.get('anchor_comparison', {})
                
                # Use actual languages from session state if available, otherwise fallback to hardcoded list
                if hasattr(st.session_state, 'languages') and st.session_state.languages:
                    all_languages = st.session_state.languages
                else:
                    all_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
                
                # Use actual dynamics languages from session state if available
                if hasattr(st.session_state, 'dynamics_languages') and st.session_state.dynamics_languages:
                    dynamics_languages = st.session_state.dynamics_languages
                else:
                    dynamics_languages = all_languages if include_anchor else [lang for lang in all_languages if lang != anchor_language]
                
                experiment_config = {
                    'anchor_language': anchor_language,
                    'include_anchor': include_anchor,
                    'dynamics_languages': dynamics_languages,
                    'comparison_languages': [anchor_language]
                }
                render_anchor_comparison_tab(comparison_metrics, experiment_config)
            else:
                st.header("🔗 Anchor Language Comparison")
                st.info("Run a simulation first to see anchor comparison results.")
                st.write("This tab will show detailed comparison metrics between the anchor language and the multilingual semantic structure.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Semantic Ising Simulator** | 
        [Documentation](https://github.com/pixiiidust/semantic-ising) | 
        [Report Issues](https://github.com/pixiiidust/semantic-ising/issues) |
        [Discussions](https://github.com/pixiiidust/semantic-ising/discussions)
        """)
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        st.error(f"Application error: {e}")
        st.info("Please check the logs for more details.")


def run_simulation_workflow(concept: str, 
                           encoder: str, 
                           T_range: List[float], 
                           anchor_language: str, 
                           include_anchor: bool,
                           concept_info: Dict[str, Any]) -> None:
    """
    Run the complete simulation workflow with progress tracking.
    
    Args:
        concept: Concept name to simulate
        encoder: Encoder model to use
        T_range: Temperature range for simulation
        anchor_language: Selected anchor language
        include_anchor: Whether anchor is included in dynamics
        concept_info: Additional information about the concept
    """
    try:
        # Import simulation functions
        from core.simulation import run_temperature_sweep
        from core.phase_detection import find_critical_temperature
        from core.post_analysis import analyze_simulation_results
        from config.validator import load_config
        
        # Load configuration
        config = load_config("config/defaults.yaml")
        sim_params = config.get('simulation_params', {})
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Loading embeddings (10%)
        status_text.text("🔄 Loading embeddings...")
        progress_bar.progress(0.1)
        
        with st.spinner("🔄 Loading embeddings..."):
            # Generate embeddings
            filename = concept_info.get('filename') if concept_info else None
            embeddings, languages = generate_embeddings(concept, encoder, filename)
            
            # Configure anchor experiment
            dynamics_languages, comparison_languages = configure_anchor_experiment(
                languages, anchor_language, include_anchor
            )
        
        # Step 2: Configuring experiment (20%)
        status_text.text("⚙️ Configuring experiment...")
        progress_bar.progress(0.2)
        
        # Extract embeddings for dynamics and anchor
        dynamics_indices = [languages.index(lang) for lang in dynamics_languages]
        dynamics_embeddings = embeddings[dynamics_indices]
        
        anchor_indices = [languages.index(lang) for lang in comparison_languages]
        anchor_embeddings = embeddings[anchor_indices]
        
        # Step 3: Temperature estimation (30%)
        status_text.text("🌡️ Estimating temperature range...")
        progress_bar.progress(0.3)
        
        # Temperature estimation already done in sidebar if auto-estimate is enabled
        # Just verify we have the range stored
        if 'estimated_range' not in st.session_state:
            try:
                estimated_tmin, estimated_tmax = estimate_practical_range(
                    dynamics_embeddings, 
                    config_max_temperature=config_defaults['temp_max']
                )
                st.session_state.estimated_range = (estimated_tmin, estimated_tmax)
                st.info(f"Auto-estimated range: {estimated_tmin:.3f} - {estimated_tmax:.3f}")
            except Exception as e:
                st.warning(f"Temperature estimation failed: {e}")
                st.session_state.estimated_range = (0.1, 5.0)
        
        # Step 4: Running temperature sweep (60%)
        status_text.text("⚙️ Running temperature sweep...")
        progress_bar.progress(0.4)
        
        # Record start time for elapsed time tracking
        simulation_start_time = time.time()
        
        # Show expected duration
        n_temps = len(T_range)
        n_sweeps = 10  # n_sweeps_per_temperature
        
        # Accurate timing estimate based on actual performance: 10.9 minutes for 200 temperatures
        # Average: ~3.3 seconds per temperature
        estimated_seconds = n_temps * 3.3
        estimated_minutes = estimated_seconds / 60
        
        if estimated_minutes >= 1:
            duration_text = f"~{estimated_minutes:.1f} minutes"
        else:
            duration_text = f"~{estimated_seconds:.0f} seconds"
            
        st.info(f"⏱️ Expected duration: {duration_text} (processing {n_temps} temperatures × {n_sweeps} sweeps each)")
        
        # Create progress callback for real-time updates
        def update_progress(progress_percent, status_message):
            # Calculate elapsed time
            elapsed_seconds = time.time() - simulation_start_time
            elapsed_minutes = elapsed_seconds / 60
            
            if elapsed_minutes >= 1:
                elapsed_text = f"{elapsed_minutes:.1f}m {elapsed_seconds % 60:.0f}s"
            else:
                elapsed_text = f"{elapsed_seconds:.0f}s"
            
            # Scale progress from 0-100 to 0.4-0.6 (temperature sweep range)
            # Convert percentage to 0.0-1.0 range
            scaled_progress = 0.4 + (progress_percent / 100.0) * 0.2  # 0.4 to 0.6
            progress_bar.progress(scaled_progress)
            status_text.text(f"{status_message} (Elapsed: {elapsed_text})")
        
        # Run simulation with config parameters and progress callback
        simulation_results = run_temperature_sweep(
            dynamics_embeddings, 
            T_range,
            store_all_temperatures=True,
            n_sweeps_per_temperature=10,  # Use reasonable number of sweeps
            sim_params=sim_params,
            progress_callback=update_progress
        )
        
        # Update status to show simulation completed
        elapsed_seconds = time.time() - simulation_start_time
        elapsed_minutes = elapsed_seconds / 60
        
        if elapsed_minutes >= 1:
            elapsed_text = f"{elapsed_minutes:.1f}m {elapsed_seconds % 60:.0f}s"
        else:
            elapsed_text = f"{elapsed_seconds:.0f}s"
            
        status_text.text(f"✅ Temperature sweep completed! (Elapsed: {elapsed_text})")
        progress_bar.progress(0.6)
        
        # Add dynamics vectors to simulation results for post-analysis
        simulation_results['dynamics_vectors'] = dynamics_embeddings
        
        # Add languages to simulation results for UMAP plotting
        simulation_results['languages'] = dynamics_languages
        
        progress_bar.progress(0.6)
        
        # Step 5: Detecting critical temperature (80%)
        status_text.text("🎯 Detecting critical temperature...")
        progress_bar.progress(0.8)
        
        # Detect critical temperature
        tc = find_critical_temperature(simulation_results)

        # Step 5.5: Rerun at Tc for entropy evolution logging
        st.info(f"🔁 Rerunning simulation at Tc = {tc:.3f} to log entropy evolution...")
        # Map config keys to simulate_at_temperature args
        sim_kwargs = {}
        if 'max_iterations' in sim_params:
            sim_kwargs['max_iter'] = sim_params['max_iterations']
        if 'convergence_threshold' in sim_params:
            # Use a much smaller convergence threshold for detailed logging
            sim_kwargs['convergence_threshold'] = 1e-8  # Much smaller than default
        if 'noise_sigma' in sim_params:
            sim_kwargs['noise_sigma'] = sim_params['noise_sigma']
        if 'update_method' in sim_params:
            sim_kwargs['update_method'] = sim_params['update_method']
        # Add any other supported keys as needed
        metrics_tc, _, convergence_info_tc = simulate_at_temperature(
            dynamics_embeddings, tc, log_every=1, **sim_kwargs
        )
        simulation_results['entropy_evolution_at_tc'] = convergence_info_tc

        # Step 6: Performing analysis (90%)
        status_text.text("📊 Performing analysis...")
        progress_bar.progress(0.9)
        
        # Step 5: Post-simulation analysis (80%)
        progress_bar.progress(0.8)
        status_text.text("Performing post-simulation analysis...")
        
        try:
            analysis_results = analyze_simulation_results(simulation_results, anchor_embeddings, tc)
            
            # Analysis completed successfully
            if analysis_results and 'anchor_comparison' in analysis_results:
                anchor_comparison = analysis_results['anchor_comparison']
                # Analysis results are available and valid
                pass
            else:
                st.warning("⚠️ Post-simulation analysis completed but no comparison results available.")
                
        except Exception as e:
            st.error(f"❌ Post-simulation analysis failed: {str(e)}")
            analysis_results = None
        
        # Step 7: Complete (100%)
        status_text.text("✅ Simulation complete!")
        progress_bar.progress(1.0)
        
        # Calculate total elapsed time
        total_elapsed_seconds = time.time() - simulation_start_time
        total_elapsed_minutes = total_elapsed_seconds / 60
        
        if total_elapsed_minutes >= 1:
            total_elapsed_text = f"{total_elapsed_minutes:.1f}m {total_elapsed_seconds % 60:.0f}s"
        else:
            total_elapsed_text = f"{total_elapsed_seconds:.0f}s"
        
        # Store results in session state
        st.session_state.simulation_results = simulation_results
        st.session_state.analysis_results = analysis_results
        st.session_state.critical_temperature = tc
        st.session_state.languages = languages
        st.session_state.dynamics_languages = dynamics_languages
        st.session_state.anchor_language = anchor_language
        st.session_state.include_anchor = include_anchor
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Simulation complete! (Elapsed: {total_elapsed_text})\nCritical temperature: {tc:.3f}")
        
    except Exception as e:
        # Clear progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        
        st.error(f"❌ Simulation failed: {str(e)}")
        logger.error(f"Simulation error: {e}", exc_info=True)


def render_overview_tab(concept: str, 
                       encoder: str, 
                       T_range: List[float], 
                       anchor_language: str, 
                       include_anchor: bool) -> None:
    """
    Render overview tab with key insights and visualizations
    
    Args:
        concept: Concept being simulated
        encoder: Encoder model used
        T_range: Temperature range
        anchor_language: Selected anchor language
        include_anchor: Whether anchor is included in dynamics
    """
    try:
        st.header("📈 Overview & Insights")
        
        # Display experiment summary
        st.subheader("🔬 Experiment Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Concept", concept.title())
            st.metric("Encoder", encoder)
        
        with col2:
            st.metric("Temperature Range", f"{T_range[0]:.1f} - {T_range[-1]:.1f}")
            st.metric("Temperature Steps", len(T_range))
        
        with col3:
            st.metric("Anchor Language", anchor_language.upper())
            st.metric("Experiment Type", "Single-phase" if include_anchor else "Two-phase")
        
        # Display key insights if results are available
        if hasattr(st.session_state, 'simulation_results') and st.session_state.simulation_results:
            st.subheader("🎯 Key Insights")
            
            simulation_results = st.session_state.simulation_results
            analysis_results = st.session_state.analysis_results
            
            # Critical temperature insight
            if hasattr(st.session_state, 'critical_temperature'):
                tc = st.session_state.critical_temperature
                st.success(f"**Critical Temperature Detected:** {tc:.3f}")
                st.write(f"The system shows a phase transition at T = {tc:.3f}, indicating the emergence of universal semantic structure.")
            
            # Anchor comparison insight
            if analysis_results and 'anchor_comparison' in analysis_results:
                comparison = analysis_results['anchor_comparison']
                cosine_similarity = comparison.get('cosine_similarity', 0.0)
                
                if cosine_similarity > 0.8:
                    st.success("**Strong Semantic Convergence**")
                    st.write(f"Cosine similarity of {cosine_similarity:.3f} indicates strong convergence between anchor and multilingual semantic structure.")
                elif cosine_similarity > 0.6:
                    st.warning("**Moderate Semantic Convergence**")
                    st.write(f"Cosine similarity of {cosine_similarity:.3f} shows moderate convergence.")
                else:
                    st.error("**Weak Semantic Convergence**")
                    st.write(f"Cosine similarity of {cosine_similarity:.3f} suggests limited convergence.")
            
            # Display overview chart
            st.subheader("📊 Entropy vs Temperature")
            if 'critical_temperature' in simulation_results:
                fig = plot_entropy_vs_temperature(simulation_results)
                st.plotly_chart(fig, use_container_width=True, key="overview_entropy_chart")
        
        else:
            st.info("Run a simulation to see insights and visualizations.")
        
        # Display methodology
        with st.expander("🔬 Methodology", expanded=False):
            st.markdown("""
            **Semantic Ising Model:**
            
            1. **Embedding Generation**: Multilingual embeddings for the target concept
            2. **Ising Dynamics**: Temperature-dependent vector updates using Metropolis/Glauber rules
            3. **Phase Detection**: Critical temperature detection using log(ξ) derivative method (knee in correlation length)
            4. **Convergence Analysis**: Convergence Summary Across Temperatures & Entropy vs Correlation Length
            5. **Comparison Analysis**: Anchor language comparison using cosine distance and similarity
            
            **Key Metrics:**
            - **Alignment**: Average cosine similarity between vectors
            - **Correlation Length**: Characteristic length scale of correlations
            - **Cosine Distance**: Primary semantic distance metric for anchor comparison (0-1, lower is better)
            - **Cosine Similarity**: Directional similarity for anchor comparison (0-1, higher is better)
            """)
        
    except Exception as e:
        logger.error(f"Error rendering overview tab: {e}")
        st.error(f"Error displaying overview: {e}")


def create_mock_simulation_results(concept: str, 
                                  encoder: str, 
                                  T_range: List[float], 
                                  anchor_language: str, 
                                  include_anchor: bool,
                                  concept_info: Dict[str, Any]) -> None:
    """
    Create mock simulation results for testing UI components.
    
    Args:
        concept: Concept name to simulate
        encoder: Encoder model to use
        T_range: Temperature range for simulation
        anchor_language: Selected anchor language
        include_anchor: Whether anchor is included in dynamics
        concept_info: Additional information about the concept
    """
    try:
        st.info("🧪 Running in TEST MODE with mock data...")
        
        # Create mock simulation results
        simulation_results = {
            'temperatures': np.array(T_range),
            'alignment': np.array([0.9 - 0.8 * np.exp(-t) for t in T_range]),
            'entropy': np.array([0.1 + 0.8 * (1 - np.exp(-t)) for t in T_range]),
            'energy': np.array([-0.8 + 0.6 * np.exp(-t) for t in T_range]),
            'correlation_length': np.array([1.2 * np.exp(-np.abs(t - 1.5)) for t in T_range]),
            'vector_snapshots': {
                T_range[len(T_range)//2]: np.random.randn(5, 768)  # Mock vectors at middle temperature
            },
            'dynamics_vectors': np.random.randn(5, 768),  # Mock dynamics vectors
            'languages': ['en', 'es', 'fr', 'de', 'it'],  # Mock languages
            'critical_temperature': 1.5
        }
        
        # Create mock analysis results
        analysis_results = {
            'critical_temperature': 1.5,
            'anchor_comparison': {
                'procrustes_distance': 0.15,
                'cka_similarity': 0.75,
                'emd_distance': 0.25,
                'kl_divergence': 0.12,
                'cosine_similarity': 0.82
            },
            'correlation_analysis': {
                'correlation_length': 1.2,
                'correlation_matrix': np.random.randn(5, 5)
            }
        }
        
        # Store results in session state
        st.session_state.simulation_results = simulation_results
        st.session_state.analysis_results = analysis_results
        st.session_state.critical_temperature = 1.5
        st.session_state.languages = ['en', 'es', 'fr', 'de', 'it']
        st.session_state.dynamics_languages = ['en', 'es', 'fr', 'de', 'it'] if include_anchor else ['es', 'fr', 'de', 'it']
        st.session_state.anchor_language = anchor_language
        st.session_state.include_anchor = include_anchor
        
        st.success(f"✅ Test simulation complete! Mock critical temperature: 1.500")
        
    except Exception as e:
        st.error(f"❌ Test simulation failed: {str(e)}")
        logger.error(f"Test simulation error: {e}", exc_info=True)


if __name__ == "__main__":
    main() 