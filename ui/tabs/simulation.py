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
    plot_convergence_history
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
            # Convergence analysis
            st.subheader("🔄 Convergence Analysis")
            
            # Debug: Show what's available in simulation_results
            st.write("**Debug Info:**")
            st.write(f"Simulation results keys: {list(simulation_results.keys())}")
            
            if 'convergence_data' in simulation_results:
                convergence_data = simulation_results['convergence_data']
                st.write(f"Convergence data found: {len(convergence_data)} temperature points")
                
                # Show sample convergence data
                if convergence_data:
                    sample_data = convergence_data[0]
                    st.write(f"Sample convergence data keys: {list(sample_data.keys())}")
                    st.write(f"Sample status: {sample_data.get('status', 'N/A')}")
                    st.write(f"Sample iterations: {sample_data.get('iterations', 'N/A')}")
                
                # Convergence summary
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary plot
                    st.write("**Debug: Creating convergence summary chart...**")
                    fig_summary = plot_convergence_history(convergence_data)
                    st.write(f"**Debug: Chart created with {len(fig_summary.data)} traces**")
                    st.plotly_chart(fig_summary, use_container_width=True, key="convergence_summary_chart")
                
                with col2:
                    # Convergence statistics
                    st.subheader("📊 Convergence Statistics")
                    
                    # Count different statuses
                    status_counts = {}
                    for data in convergence_data:
                        status = data['status']
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    for status, count in status_counts.items():
                        color = {
                            'converged': 'green',
                            'plateau': 'orange', 
                            'diverging': 'red',
                            'max_steps': 'purple',
                            'error': 'gray'
                        }.get(status, 'blue')
                        
                        st.metric(f"{status.title()}", count, delta=None)
                    
                    # Average iterations
                    valid_iterations = [data['iterations'] for data in convergence_data 
                                      if data['status'] != 'error' and data['iterations'] > 0]
                    if valid_iterations:
                        avg_iterations = sum(valid_iterations) / len(valid_iterations)
                        st.metric("Average Iterations", f"{avg_iterations:.1f}")
                
                # Detailed convergence for specific temperature
                st.subheader("🔍 Detailed Convergence")
                
                if 'temperatures' in simulation_results:
                    temps = simulation_results['temperatures']
                    selected_temp = st.selectbox(
                        "Select temperature for detailed convergence analysis:",
                        temps,
                        format_func=lambda x: f"T = {x:.3f}"
                    )
                    
                    fig_detail = plot_convergence_history(convergence_data, selected_temp)
                    st.plotly_chart(fig_detail, use_container_width=True, key="convergence_detail_chart")
                    
                    # Show convergence info for selected temperature
                    temp_data = next((data for data in convergence_data 
                                    if abs(data['temperature'] - selected_temp) < 1e-6), None)
                    
                    if temp_data and temp_data['convergence_infos']:
                        conv_info = temp_data['convergence_infos'][0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status", temp_data['status'].title())
                        with col2:
                            st.metric("Final Difference", f"{temp_data['final_diff']:.2e}")
                        with col3:
                            st.metric("Iterations", temp_data['iterations'])
                        
                        # Show convergence history as table
                        if conv_info['diff_history']:
                            st.subheader("📋 Convergence History")
                            history_data = {
                                'Iteration': conv_info['logged_steps'],
                                'Difference': [f"{d:.2e}" for d in conv_info['diff_history']],
                                'Alignment': [f"{a:.4f}" for a in conv_info['alignment_history']]
                            }
                            st.dataframe(history_data, use_container_width=True)
            else:
                st.warning("No convergence data available in simulation results.")
                st.write("**Expected convergence_data structure:**")
                st.write("""
                ```python
                convergence_data = [
                    {
                        'temperature': float,
                        'convergence_infos': [{'diff_history': [...], 'logged_steps': [...], ...}],
                        'final_diff': float,
                        'status': str,  # 'converged', 'plateau', 'diverging', 'max_steps', 'error'
                        'iterations': int
                    },
                    ...
                ]
                ```""")
        
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
            fig = plot_entropy_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="entropy_tab_chart")
        
        with tab2:
            st.subheader("Alignment vs Temperature")
            fig = plot_alignment_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="alignment_tab_chart")
        
        with tab3:
            st.subheader("Energy vs Temperature")
            fig = plot_energy_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="energy_tab_chart")
        
        with tab4:
            st.subheader("Correlation Length vs Temperature")
            fig = plot_correlation_length_vs_temperature(simulation_results)
            st.plotly_chart(fig, use_container_width=True, key="correlation_length_tab_chart")
        
        # Display anchor comparison if available
        if analysis_results and 'anchor_comparison' in analysis_results:
            st.subheader("🔗 Anchor Comparison Results")
            render_anchor_comparison_summary(analysis_results['anchor_comparison'])
            
            # Display interpretation
            display_anchor_interpretation(analysis_results['anchor_comparison'])
        
        # Display power law analysis if available
        if analysis_results and 'power_law_analysis' in analysis_results:
            st.subheader("📈 Power Law Analysis")
            from ui.components import render_power_law_summary
            render_power_law_summary(analysis_results['power_law_analysis'])
        
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
        
        cka_similarity = comparison_metrics.get('cka_similarity', 0.0)
        procrustes_distance = comparison_metrics.get('procrustes_distance', 1.0)
        cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
        
        # Determine overall similarity assessment
        if cka_similarity > 0.7 and procrustes_distance < 0.3 and cosine_similarity > 0.8:
            st.success("**Strong semantic similarity detected**")
            interpretation = """
            The anchor language shows strong alignment with the multilingual semantic structure.
            This suggests that the anchor language shares similar semantic representations
            with other languages in the concept space.
            """
        elif cka_similarity > 0.4 and procrustes_distance < 0.6 and cosine_similarity > 0.6:
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
                "CKA Similarity",
                f"{cka_similarity:.3f}",
                help="Centered Kernel Alignment similarity (0-1 scale)"
            )
            st.metric(
                "Procrustes Distance",
                f"{procrustes_distance:.3f}",
                help="Structural alignment distance (lower is better)"
            )
        
        with col2:
            st.metric(
                "Cosine Similarity",
                f"{cosine_similarity:.3f}",
                help="Average cosine similarity between vectors"
            )
            st.metric(
                "EMD Distance",
                f"{comparison_metrics.get('emd_distance', 0.0):.3f}",
                help="Earth Mover's Distance (lower is better)"
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