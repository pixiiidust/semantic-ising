"""
Anchor comparison tab for Streamlit interface (Phase 9)
Provides detailed analysis of anchor language comparison results
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

# Import UI components
from ui.components import (
    render_anchor_comparison_summary,
    render_error_message,
    render_warning_message,
    render_success_message
)

# Import chart functions
from ui.charts import plot_full_umap_projection

logger = logging.getLogger(__name__)


def render_anchor_comparison_tab(comparison_metrics: Dict[str, float], experiment_config: Dict[str, Any]):
    """
    Render anchor comparison tab
    """
    st.header("🔗 Anchor Language Comparison")
    
    if not comparison_metrics:
        st.warning("No comparison metrics available. Run simulation first.")
        return
    
    try:
        # Display Critical Temperature and main metrics in 3-column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(st.session_state, 'critical_temperature') and st.session_state.critical_temperature:
                tc = st.session_state.critical_temperature
                st.metric("Critical Temperature", f"{tc:.3f}", help="Critical temperature, Tc, an estimate of where phase transition occurs")
            else:
                st.metric("Critical Temperature", "N/A", help="Critical temperature not available")
        
        with col2:
            st.metric("Cosine Distance", f"{comparison_metrics['cosine_distance']:.4f}", help="Distance between anchor language vector and multilingual meta-vector at Tc (0-1, lower is better)")
        
        with col3:
            st.metric("Cosine Similarity", f"{comparison_metrics['cosine_similarity']:.4f}", help="Similarity between anchor language vector and multilingual meta-vector at Tc (0-1, higher is better)")
        
        # Experiment configuration in expander
        with st.expander("⚙️ Experiment Configuration", expanded=False):
            if experiment_config:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Anchor Language:** {experiment_config.get('anchor_language', 'Unknown')}")
                    st.write(f"**Include Anchor:** {'Yes' if experiment_config.get('include_anchor', False) else 'No'}")
                
                with col2:
                    dynamics_langs = experiment_config.get('dynamics_languages', [])
                    st.write(f"**Dynamics Languages:** {len(dynamics_langs)} languages")
                    if dynamics_langs:
                        st.write(f"Languages: {', '.join(dynamics_langs[:5])}{'...' if len(dynamics_langs) > 5 else ''}")
        
        # Interpretation in expander
        with st.expander("💡 Interpretation", expanded=True):
            if comparison_metrics['cosine_similarity'] > 0.8:
                st.success("Strong semantic similarity detected")
            elif comparison_metrics['cosine_similarity'] > 0.6:
                st.warning("Moderate semantic similarity")
            else:
                st.error("Weak semantic similarity")
            
            st.write("**What's being compared:**")
            st.write("""
            * Cosine similarity and distance quantify how closely the anchor language aligns with the emergent multilingual structure at Tc."
            * The emergent multilingual structure is estimated by a **meta-vector** (centroid) computed as the average of all language vectors at the critical temperature.
            * Computation: **Meta-vector** = mean(all language vectors at Tc), then normalized to unit length.""")
        
        # UMAP projection if available
        if hasattr(st.session_state, 'simulation_results') and hasattr(st.session_state, 'analysis_results'):
            st.subheader("🗺️ UMAP Projection")

            simulation_results = st.session_state.simulation_results
            analysis_results = st.session_state.analysis_results
            vector_snapshots = simulation_results.get('vector_snapshots', {})
            snapshot_dir = simulation_results.get('snapshot_directory')
            available_snapshot_temps = simulation_results.get('available_snapshot_temperatures', [])
            selected_temp = None

            # Add slider if multiple snapshots are available
            if (snapshot_dir and available_snapshot_temps and len(available_snapshot_temps) > 1) or (vector_snapshots and len(vector_snapshots) > 1):
                if snapshot_dir and available_snapshot_temps:
                    # Use disk-based snapshots
                    snapshot_temps = available_snapshot_temps
                else:
                    # Use memory-based snapshots
                    snapshot_temps = sorted(list(vector_snapshots.keys()))
                
                tc = analysis_results.get('critical_temperature')
                
                # Find the temperature in snapshots closest to Tc to be the default
                default_temp = min(snapshot_temps, key=lambda t: abs(t - tc)) if tc else snapshot_temps[0]

                selected_temp = st.slider(
                    "Select Temperature for UMAP",
                    min_value=min(snapshot_temps),
                    max_value=max(snapshot_temps),
                    value=default_temp,
                    step=snapshot_temps[1] - snapshot_temps[0] if len(snapshot_temps) > 1 else 0.01,
                    format="%.3f",
                    help="Scrub through temperatures to see how semantic structure evolves. The plot dynamically updates."
                )

            try:
                # Extract anchor language information from experiment config
                anchor_language = experiment_config.get('anchor_language') if experiment_config else None
                include_anchor = experiment_config.get('include_anchor', False) if experiment_config else False
                
                fig = plot_full_umap_projection(
                    simulation_results, 
                    analysis_results,
                    anchor_language=anchor_language,
                    include_anchor=include_anchor,
                    selected_temperature=selected_temp
                )
                st.plotly_chart(fig, use_container_width=True, key="umap_projection_chart")
            except Exception as e:
                st.info("UMAP projection not available for current data")
        
    except Exception as e:
        logger.error(f"Error displaying comparison metrics: {e}")
        st.error("Error displaying comparison metrics. Please try again.")


def display_comparison_metrics(comparison_metrics: Dict[str, float]) -> None:
    """
    Display comparison metrics in organized format
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        st.subheader("📊 Comparison Metrics")
        
        # Primary metric: Cosine Distance
        st.metric(
            "Cosine Distance",
            f"{comparison_metrics.get('cosine_distance', 0.0):.4f}",
            help="Distance between anchor language vector and multilingual meta-vector at Tc. Lower values indicate more similar meaning."
        )
        
        # Supporting metric: Cosine Similarity
        st.metric(
            "Cosine Similarity",
            f"{comparison_metrics.get('cosine_similarity', 0.0):.4f}",
            help="Similarity between anchor language vector and multilingual meta-vector at Tc. Higher values indicate more similar vector directions."
        )
        
    except Exception as e:
        logger.error(f"Error displaying comparison metrics: {e}")
        render_error_message(e, "comparison metrics")


def display_experiment_config(experiment_config: Dict[str, Any]) -> None:
    """
    Display experiment configuration details
    
    Args:
        experiment_config: Experiment configuration dictionary
    """
    try:
        st.subheader("🔬 Experiment Configuration")
        
        if not experiment_config:
            st.info("No experiment configuration available")
            return
        
        # Create columns for configuration details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Anchor Language:** {experiment_config.get('anchor_language', 'N/A')}")
            st.write(f"**Include Anchor:** {experiment_config.get('include_anchor', 'N/A')}")
        
        with col2:
            dynamics_languages = experiment_config.get('dynamics_languages', [])
            comparison_languages = experiment_config.get('comparison_languages', [])
            st.write(f"**Dynamics Languages:** {len(dynamics_languages)} languages")
            st.write(f"**Comparison Languages:** {len(comparison_languages)} languages")
        
        # Display experiment type
        include_anchor = experiment_config.get('include_anchor', False)
        experiment_type = "Single-phase" if include_anchor else "Two-phase"
        st.info(f"**Experiment Type:** {experiment_type}")
        
        # Display detailed language lists
        with st.expander("🌍 Language Details", expanded=False):
            if dynamics_languages:
                st.write("**Languages in Dynamics:**")
                st.write(", ".join(dynamics_languages))
            
            if comparison_languages:
                st.write("**Languages in Comparison:**")
                st.write(", ".join(comparison_languages))
        
    except Exception as e:
        logger.error(f"Error displaying experiment config: {e}")
        render_error_message(e, "experiment config")


def display_detailed_comparison_analysis(comparison_metrics: Dict[str, float]) -> None:
    """
    Display detailed analysis of comparison metrics
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        st.subheader("🔍 Detailed Analysis")
        
        # Create tabs for different analysis aspects
        tab1, tab2, tab3 = st.tabs(["Metric Breakdown", "Statistical Analysis", "Visualization"])
        
        with tab1:
            display_metric_breakdown(comparison_metrics)
        
        with tab2:
            display_statistical_analysis(comparison_metrics)
        
        with tab3:
            display_comparison_visualization(comparison_metrics)
        
    except Exception as e:
        logger.error(f"Error displaying detailed analysis: {e}")
        render_error_message(e, "detailed analysis")


def display_metric_breakdown(comparison_metrics: Dict[str, float]) -> None:
    """
    Display detailed breakdown of each metric
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        # Define metric descriptions and thresholds
        metric_info = {
            'cosine_distance': {
                'description': 'Distance between anchor language vector and multilingual meta-vector (centroid) at critical temperature',
                'excellent': '< 0.1',
                'good': '0.1 - 0.3',
                'moderate': '0.3 - 0.6',
                'poor': '> 0.6'
            },
            'cosine_similarity': {
                'description': 'Similarity between anchor language vector and multilingual meta-vector (centroid) at critical temperature',
                'excellent': '> 0.9',
                'good': '0.7 - 0.9',
                'moderate': '0.5 - 0.7',
                'poor': '< 0.5'
            }
        }
        
        for metric_name, value in comparison_metrics.items():
            if metric_name in metric_info:
                info = metric_info[metric_name]
                
                st.write(f"### {metric_name.replace('_', ' ').title()}")
                st.write(f"**Value:** {value:.4f}")
                st.write(f"**Description:** {info['description']}")
                
                # Determine quality level
                if metric_name in ['cosine_similarity']:
                    # Higher is better for these metrics
                    if value > 0.8:
                        quality = "excellent"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif value > 0.6:
                        quality = "good"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif value > 0.4:
                        quality = "moderate"
                        st.warning(f"**Quality:** {quality} ({info[quality]})")
                    else:
                        quality = "poor"
                        st.error(f"**Quality:** {quality} ({info[quality]})")
                else:
                    # Lower is better for these metrics
                    if value < 0.1:
                        quality = "excellent"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif value < 0.3:
                        quality = "good"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif value < 0.6:
                        quality = "moderate"
                        st.warning(f"**Quality:** {quality} ({info[quality]})")
                    else:
                        quality = "poor"
                        st.error(f"**Quality:** {quality} ({info[quality]})")
                
                st.divider()
        
    except Exception as e:
        logger.error(f"Error displaying metric breakdown: {e}")
        render_error_message(e, "metric breakdown")


def display_statistical_analysis(comparison_metrics: Dict[str, float]) -> None:
    """
    Display statistical analysis of comparison metrics
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        # Calculate composite scores
        cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
        
        # Calculate different composite scores
        directional_score = cosine_similarity
        
        st.write("### Composite Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Directional Similarity", f"{directional_score:.3f}")
        
        with col2:
            st.metric("Overall Similarity", f"{directional_score:.3f}")
        
        # Display score interpretation
        st.write("### Score Interpretation")
        
        if directional_score > 0.8:
            st.success("**Excellent semantic alignment** - Strong evidence of shared semantic space")
        elif directional_score > 0.6:
            st.success("**Good semantic alignment** - Clear evidence of shared semantic space")
        elif directional_score > 0.4:
            st.warning("**Moderate semantic alignment** - Some shared semantic space")
        else:
            st.error("**Weak semantic alignment** - Limited shared semantic space")
        
        # Display metric correlations
        st.write("### Metric Analysis")
        
        # Check for consistency across metrics
        high_similarity_metrics = []
        low_similarity_metrics = []
        
        for metric_name, value in comparison_metrics.items():
            if metric_name in ['cosine_similarity']:
                if value > 0.7:
                    high_similarity_metrics.append(metric_name)
                elif value < 0.4:
                    low_similarity_metrics.append(metric_name)
        
        if high_similarity_metrics:
            st.success(f"**Strong metrics:** {', '.join(high_similarity_metrics)}")
        
        if low_similarity_metrics:
            st.error(f"**Weak metrics:** {', '.join(low_similarity_metrics)}")
        
    except Exception as e:
        logger.error(f"Error displaying statistical analysis: {e}")
        render_error_message(e, "statistical analysis")


def display_comparison_visualization(comparison_metrics: Dict[str, float]) -> None:
    """
    Display visualization of comparison metrics
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        # Create radar chart of metrics
        import plotly.graph_objects as go
        
        # Prepare data for radar chart
        metric_names = list(comparison_metrics.keys())
        values = list(comparison_metrics.values())
        
        # Normalize values to 0-1 scale for radar chart
        normalized_values = []
        for i, (name, value) in enumerate(zip(metric_names, values)):
            if name in ['cosine_similarity']:
                # Higher is better, keep as is
                normalized_values.append(min(value, 1.0))
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metric_names,
            fill='toself',
            name='Similarity Scores',
            line_color='#1f77b4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparison Metrics Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="radar_chart")
        
        # Display metric values table
        st.write("### Metric Values")
        
        data = {
            'Metric': [name.replace('_', ' ').title() for name in metric_names],
            'Value': [f"{value:.4f}" for value in values],
            'Normalized': [f"{norm:.3f}" for norm in normalized_values]
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying comparison visualization: {e}")
        render_error_message(e, "comparison visualization")


def display_umap_projection(experiment_config: Dict[str, Any] = None) -> None:
    """
    Display UMAP projection if available
    """
    try:
        if (hasattr(st.session_state, 'simulation_results') and 
            hasattr(st.session_state, 'analysis_results')):
            
            simulation_results = st.session_state.simulation_results
            analysis_results = st.session_state.analysis_results
            
            if (simulation_results and analysis_results and 
                'vector_snapshots' in simulation_results):
                
                st.subheader("🗺️ UMAP Projection")
                
                # Extract anchor language information from experiment config
                anchor_language = experiment_config.get('anchor_language') if experiment_config else None
                include_anchor = experiment_config.get('include_anchor', False) if experiment_config else False
                
                fig = plot_full_umap_projection(
                    simulation_results, 
                    analysis_results,
                    anchor_language=anchor_language,
                    include_anchor=include_anchor
                )
                if fig.data:  # Check if figure has data
                    st.plotly_chart(fig, use_container_width=True, key="umap_projection_display_chart")
                else:
                    st.info("UMAP projection not available (vector snapshots may not be stored)")
            else:
                st.info("UMAP projection requires vector snapshots from simulation")
        else:
            st.info("Run simulation first to see UMAP projection")
            
    except Exception as e:
        logger.error(f"Error displaying UMAP projection: {e}")
        render_error_message(e, "UMAP projection")


def display_interpretation_and_insights(comparison_metrics: Dict[str, float],
                                       experiment_config: Dict[str, Any]) -> None:
    """
    Display interpretation and insights from comparison results
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
        experiment_config: Experiment configuration
    """
    try:
        st.subheader("🧠 Interpretation & Insights")
        
        # Calculate overall assessment
        cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
        
        # Determine experiment type
        include_anchor = experiment_config.get('include_anchor', False)
        anchor_language = experiment_config.get('anchor_language', 'Unknown')
        
        if include_anchor:
            experiment_type = "Single-phase"
            question = f"Does {anchor_language} share semantic space with other languages?"
        else:
            experiment_type = "Two-phase"
            question = f"How does {anchor_language} compare to multilingual semantic dynamics?"
        
        # Display experiment context
        st.write(f"**Experiment Type:** {experiment_type}")
        st.write(f"**Research Question:** {question}")
        
        # Display insights based on composite score
        if cosine_similarity > 0.7:
            st.success("**Strong Semantic Alignment Detected**")
            insights = f"""
            **Key Insights:**
            
            • The {anchor_language} language shows strong alignment with the multilingual semantic structure
            • This suggests shared semantic representations across languages for this concept
            • The high similarity across multiple metrics indicates robust semantic convergence
            • This supports the hypothesis of universal semantic structures
            
            **Scientific Implications:**
            
            • Evidence for cross-lingual semantic universals
            • Suggests similar cognitive representations across languages
            • Supports the existence of shared conceptual spaces
            """
        elif cosine_similarity > 0.5:
            st.warning("**Moderate Semantic Alignment**")
            insights = f"""
            **Key Insights:**
            
            • The {anchor_language} language shows moderate alignment with the multilingual semantic structure
            • There is some shared semantic space, but differences remain
            • This suggests partial semantic convergence across languages
            • Cultural or linguistic factors may influence semantic representations
            
            **Scientific Implications:**
            
            • Mixed evidence for semantic universals
            • Suggests both shared and language-specific semantic features
            • Indicates the need for more nuanced cross-lingual analysis
            """
        else:
            st.error("**Weak Semantic Alignment**")
            insights = f"""
            **Key Insights:**
            
            • The {anchor_language} language shows weak alignment with the multilingual semantic structure
            • This suggests significant differences in semantic representations across languages
            • Cultural or linguistic factors strongly influence semantic space organization
            • Limited evidence for universal semantic structures
            
            **Scientific Implications:**
            
            • Suggests language-specific semantic organization
            • Indicates strong cultural influences on conceptual representations
            • Challenges the universality hypothesis for this concept
            """
        
        st.write(insights)
        
        # Display methodological insights
        display_methodological_insights(comparison_metrics, experiment_config)
        
    except Exception as e:
        logger.error(f"Error displaying interpretation and insights: {e}")
        render_error_message(e, "interpretation and insights")


def display_methodological_insights(comparison_metrics: Dict[str, float],
                                   experiment_config: Dict[str, Any]) -> None:
    """
    Display methodological insights and recommendations
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
        experiment_config: Experiment configuration
    """
    try:
        st.subheader("🔬 Methodological Insights")
        
        # Analyze metric consistency
        metric_values = list(comparison_metrics.values())
        metric_range = max(metric_values) - min(metric_values)
        
        if metric_range < 0.3:
            consistency = "High"
            consistency_color = "success"
        elif metric_range < 0.6:
            consistency = "Moderate"
            consistency_color = "warning"
        else:
            consistency = "Low"
            consistency_color = "error"
        
        st.write(f"**Metric Consistency:** {consistency}")
        
        # Display recommendations
        st.write("**Recommendations:**")
        
        if consistency == "High":
            st.success("• Results are consistent across metrics - high confidence in findings")
            st.success("• Consider expanding to more concepts and languages")
            st.success("• Investigate underlying mechanisms of semantic convergence")
        elif consistency == "Moderate":
            st.warning("• Some metric disagreement - consider methodological refinements")
            st.warning("• Investigate why certain metrics show different patterns")
            st.warning("• Consider additional validation methods")
        else:
            st.error("• High metric disagreement - results may be unreliable")
            st.error("• Review experimental design and data quality")
            st.error("• Consider alternative analysis approaches")
        
        # Display experimental design insights
        include_anchor = experiment_config.get('include_anchor', False)
        
        if include_anchor:
            st.write("**Single-phase Design Insights:**")
            st.write("• Anchor language participates in dynamics, providing direct comparison")
            st.write("• Results show how anchor integrates with multilingual semantic space")
            st.write("• Useful for understanding shared semantic structures")
        else:
            st.write("**Two-phase Design Insights:**")
            st.write("• Anchor language compared to emergent multilingual structure")
            st.write("• Results show how anchor relates to collective semantic dynamics")
            st.write("• Useful for understanding cross-lingual semantic relationships")
        
    except Exception as e:
        logger.error(f"Error displaying methodological insights: {e}")
        render_error_message(e, "methodological insights") 