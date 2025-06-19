"""
Simulation tab for Streamlit interface (Phase 9)
Handles main simulation interface with anchor configuration
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any
import logging

# Import UI components
from ui.components import (
    render_experiment_description,
    render_simulation_progress,
    render_critical_temperature_display,
    render_anchor_comparison_summary,
    render_error_message,
    render_success_message
)

# Import chart functions
from ui.charts import (
    plot_entropy_vs_temperature,
    plot_alignment_vs_temperature,
    plot_energy_vs_temperature,
    plot_correlation_length_vs_temperature,
    plot_convergence_history,
    plot_convergence_summary,
    plot_entropy_vs_correlation_length
)

logger = logging.getLogger(__name__)


def render_simulation_tab(concept: str, 
                         encoder: str, 
                         T_range: List[float], 
                         anchor_language: str, 
                         include_anchor: bool) -> None:
    """
    Render simulation tab with results and convergence analysis.
    
    Args:
        concept: Concept name to simulate
        encoder: Encoder model to use
        T_range: Temperature range for simulation
        anchor_language: Selected anchor language
        include_anchor: Whether anchor is included in dynamics
    """
    st.header("⚙️ Simulation Results")
    
    # Check if simulation results are available
    if not hasattr(st.session_state, 'simulation_results') or not st.session_state.simulation_results:
        st.warning("No simulation results available. Please run a simulation first.")
        return
    
    simulation_results = st.session_state.simulation_results
    
    # Display basic results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
            st.metric("Critical Temperature", f"{tc:.3f}")
        else:
            st.metric("Critical Temperature", "Not detected")
    
    with col2:
        if 'temperatures' in simulation_results:
            n_temps = len(simulation_results['temperatures'])
            st.metric("Temperature Points", n_temps)
        else:
            st.metric("Temperature Points", "N/A")
    
    with col3:
        if hasattr(st.session_state, 'languages'):
            n_langs = len(st.session_state.languages)
            st.metric("Languages", n_langs)
        else:
            st.metric("Languages", "N/A")
    
    # Display main metrics plots
    st.subheader("📊 Simulation Metrics")
    
    if 'temperatures' in simulation_results and 'alignment' in simulation_results:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🔄 Convergence", "📋 Details"])
        
        with tab1:
            # Main metrics plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Alignment plot
                fig_alignment = plot_alignment_vs_temperature(simulation_results)
                st.plotly_chart(fig_alignment, use_container_width=True, key="alignment_chart")
            
            with col2:
                # Entropy plot
                fig_entropy = plot_entropy_vs_temperature(simulation_results)
                st.plotly_chart(fig_entropy, use_container_width=True, key="entropy_chart")
            
            # Energy and correlation length
            col3, col4 = st.columns(2)
            
            with col3:
                fig_energy = plot_energy_vs_temperature(simulation_results)
                st.plotly_chart(fig_energy, use_container_width=True, key="energy_chart")
            
            with col4:
                fig_corr = plot_correlation_length_vs_temperature(simulation_results)
                st.plotly_chart(fig_corr, use_container_width=True, key="correlation_length_chart")
        
        with tab2:
            st.subheader("🔄 Convergence Analysis")
            # Convergence analysis
            if 'convergence_data' in simulation_results:
                convergence_data = simulation_results['convergence_data']
                # Show convergence summary across temperatures
                tc = st.session_state.critical_temperature if hasattr(st.session_state, 'critical_temperature') else None
                st.markdown("<h4>Convergence Summary Across Temperatures:</h4>", unsafe_allow_html=True)
                st.write("""
                * This chart shows how well the simulation converged at each temperature point. 
                * Green markers indicate successful convergence, orange shows plateau behavior, and red indicates divergence. 
                * The vertical red dashed line marks the critical temperature (Tc), where the system transitions from order to disorder.""")
                fig_summary = plot_convergence_summary(convergence_data, tc=tc)
                st.plotly_chart(fig_summary, use_container_width=True)
                
                # Show entropy vs correlation length instead of entropy evolution at Tc
                st.markdown("<h4>Entropy vs Correlation Length:</h4>", unsafe_allow_html=True)
                st.write("""
                * This chart shows how system disorder (entropy) and the range of semantic correlations (correlation length) evolve together as temperature changes. 
                * When both entropy rises and correlation length collapses, the system loses long-range order and fails to converge, marking the phase transition. 
                * The critical temperature point is highlighted, showing where the system transitions from ordered to disordered behavior.""")
                fig_entropy_corr = plot_entropy_vs_correlation_length(simulation_results)
                st.plotly_chart(fig_entropy_corr, use_container_width=True)
                
                # Add explanation
                st.info("""
                **Convergence Analysis Explanation:**
                - **Summary Chart**: Shows convergence status (green=converged, orange=plateau, red=diverging) and final difference values across all temperatures
                - **Entropy vs Correlation Length**: Shows the relationship between disorder and spatial correlation at each temperature, with the critical temperature (Tc) highlighted
                - **Critical Temperature**: The point where the system transitions from ordered to disordered behavior, marked on all charts
                """)
            else:
                st.warning("No convergence data available in simulation results.")
        
        with tab3:
            # Detailed information
            st.subheader("📋 Simulation Details")
            
            # Experiment configuration
            st.write("**Experiment Configuration:**")
            config_data = {
                "Concept": concept,
                "Encoder": encoder,
                "Temperature Range": f"{T_range[0]:.3f} - {T_range[-1]:.3f}",
                "Temperature Steps": len(T_range),
                "Anchor Language": anchor_language,
                "Include Anchor": "Yes" if include_anchor else "No",
                "Experiment Type": "Single-phase" if include_anchor else "Two-phase"
            }
            
            for key, value in config_data.items():
                st.write(f"- **{key}:** {value}")
            
            # Languages used
            if hasattr(st.session_state, 'languages'):
                st.write(f"**Languages ({len(st.session_state.languages)}):**")
                st.write(", ".join(st.session_state.languages))
            
            # Simulation parameters
            st.write("**Simulation Parameters:**")
            param_data = {
                "Max Iterations": "1000",
                "Convergence Threshold": "1e-3",
                "Log Every": "50 steps",
                "Slope Tolerance": "5e-4",
                "Plateau Patience": "3",
                "Diverge Tolerance": "0.05"
            }
            
            for key, value in param_data.items():
                st.write(f"- **{key}:** {value}")
    else:
        st.error("Invalid simulation results format.")


def display_simulation_results() -> None:
    """
    Display simulation results if available in session state
    """
    try:
        if not hasattr(st.session_state, 'simulation_results') or st.session_state.simulation_results is None:
            return
        
        simulation_results = st.session_state.simulation_results
        analysis_results = st.session_state.analysis_results
        tc = st.session_state.critical_temperature
        
        # Add critical temperature to simulation results for plotting
        simulation_results['critical_temperature'] = tc
        
        st.subheader("📊 Simulation Results")
        
        # Display critical temperature
        render_critical_temperature_display(tc)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Entropy", "Alignment", "Energy", "Correlation Length"])
        
        with tab1:
            st.subheader("Entropy vs Temperature")
            st.write("This chart shows how system disorder (entropy) changes with temperature. At low temperatures, the system is ordered (low entropy). As temperature increases, disorder grows until reaching a maximum at the critical temperature, where the system transitions from ordered to disordered behavior.")
            fig = plot_entropy_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="entropy_tab_chart")
        
        with tab2:
            st.subheader("Alignment vs Temperature")
            st.write("This chart shows how well the multilingual vectors align with each other as temperature changes. High alignment indicates strong semantic convergence across languages. The critical temperature marks where alignment begins to break down, signaling the loss of universal semantic structure.")
            fig = plot_alignment_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="alignment_tab_chart")
        
        with tab3:
            st.subheader("Energy vs Temperature")
            st.write("This chart shows the system's energy as temperature increases. At low temperatures, the system minimizes energy through strong semantic coupling. As temperature rises, thermal fluctuations overcome the coupling, leading to higher energy states and eventual disorder.")
            fig = plot_energy_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="energy_tab_chart")
        
        with tab4:
            st.subheader("Correlation Length vs Temperature")
            st.write("This chart shows how far semantic correlations extend in the system. At low temperatures, correlations extend far (high correlation length). As temperature approaches the critical point, correlations become shorter-ranged, indicating the breakdown of long-range semantic order.")
            fig = plot_correlation_length_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="correlation_length_tab_chart")
        
        # Display anchor comparison if available
        if analysis_results and 'anchor_comparison' in analysis_results:
            st.subheader("🔗 Anchor Comparison Results")
            render_anchor_comparison_summary(analysis_results['anchor_comparison'])
            
            # Display interpretation
            display_anchor_interpretation(analysis_results['anchor_comparison'])
        
    except Exception as e:
        logger.error(f"Error displaying simulation results: {e}")
        render_error_message(e, "displaying results")


def display_anchor_interpretation(comparison_metrics: Dict[str, float]) -> None:
    """
    Display interpretation of anchor comparison metrics
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        st.subheader("🔍 Interpretation")
        
        cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
        cosine_distance = comparison_metrics.get('cosine_distance', 1.0)
        
        # Determine overall similarity assessment based on cosine similarity
        if cosine_similarity > 0.8:
            st.success("**Strong semantic similarity detected**")
            interpretation = """
            The anchor language shows strong alignment with the multilingual semantic structure.
            This suggests that the anchor language shares similar semantic representations
            with other languages in the concept space.
            """
        elif cosine_similarity > 0.6:
            st.warning("**Moderate semantic similarity**")
            interpretation = """
            The anchor language shows moderate alignment with the multilingual semantic structure.
            There is some shared semantic space, but differences remain.
            """
        else:
            st.error("**Weak semantic similarity**")
            interpretation = """
            The anchor language shows weak alignment with the multilingual semantic structure.
            This suggests significant differences in semantic representations across languages.
            """
        
        st.write(interpretation)
        
        # Detailed metric breakdown
        st.subheader("📋 Metric Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Cosine Distance",
                f"{cosine_distance:.3f}",
                help="Distance between anchor language vector and multilingual meta-vector at Tc (0-1, lower is better)"
            )
        
        with col2:
            st.metric(
                "Cosine Similarity",
                f"{cosine_similarity:.3f}",
                help="Similarity between anchor language vector and multilingual meta-vector at Tc (0-1, higher is better)"
            )
        
    except Exception as e:
        logger.error(f"Error displaying anchor interpretation: {e}")
        render_error_message(e, "anchor interpretation")


def clear_simulation_results() -> None:
    """
    Clear simulation results from session state
    """
    try:
        if hasattr(st.session_state, 'simulation_results'):
            del st.session_state.simulation_results
        if hasattr(st.session_state, 'analysis_results'):
            del st.session_state.analysis_results
        if hasattr(st.session_state, 'critical_temperature'):
            del st.session_state.critical_temperature
        if hasattr(st.session_state, 'estimated_range'):
            del st.session_state.estimated_range
        st.success("Simulation results cleared successfully!")
    except Exception as e:
        logger.error(f"Error clearing simulation results: {e}")
        st.error("Failed to clear simulation results") 