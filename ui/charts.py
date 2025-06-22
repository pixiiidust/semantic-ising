"""
Chart generation functions for UI visualization (Phase 9)
Creates interactive Plotly charts for simulation results
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List
import logging
import streamlit as st
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import linregress
from core.clustering import cluster_vectors_kmeans, cluster_vectors, ising_compatible_clustering
import pandas as pd

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'az': 'Azerbaijani',
    'as': 'Assamese',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'el': 'Greek',
    'et': 'Estonian',
    'fa': 'Persian',
    'fi': 'Finnish',
    'ga': 'Irish',
    'gl': 'Galician',
    'gu': 'Gujarati',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'ka': 'Georgian',
    'ko': 'Korean',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ku': 'Kurdish',
    'ky': 'Kyrgyz',
    'la': 'Latin',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'my': 'Burmese',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'pa': 'Punjabi',
    'pl': 'Polish',
    'ro': 'Romanian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'su': 'Sundanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tl': 'Tagalog',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zu': 'Zulu',
}


def safe_plotly_chart(fig, title, key):
    if fig is None or (hasattr(fig, 'data') and len(fig.data) == 0):
        st.warning(f"No valid data for {title}.")
        return
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_entropy_vs_temperature(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot entropy vs temperature with Tc marker
    
    Args:
        simulation_results: Dictionary containing metrics and critical temperature
        
    Returns:
        plotly.graph_objects.Figure: Interactive entropy vs temperature plot
    """
    try:
        # Check if required data exists
        if 'temperatures' not in simulation_results or 'entropy' not in simulation_results:
            logger.warning("Missing 'temperatures' or 'entropy' in simulation_results")
            return go.Figure()
        
        temperatures = simulation_results['temperatures']
        entropy = simulation_results['entropy']
        
        # Check for valid data
        if len(temperatures) == 0 or len(entropy) == 0:
            logger.warning("Empty temperature or entropy data")
            return go.Figure()
        
        # Filter out NaN values
        valid_mask = np.isfinite(temperatures) & np.isfinite(entropy)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) data points for entropy plot")
            return go.Figure()
        
        valid_temps = temperatures[valid_mask]
        valid_entropy = entropy[valid_mask]
        
        # Create the main plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_temps,
            y=valid_entropy,
            mode='lines+markers',
            name='Entropy',
            line=dict(color='#1f77b4', width=1.5),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            try:
                tc_formatted = f"Tc = {float(tc):.3f}"
            except (ValueError, TypeError):
                tc_formatted = f"Tc = {tc}"
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=tc_formatted,
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title="Entropy vs Temperature",
            xaxis_title="Temperature",
            yaxis_title="Entropy",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating entropy plot: {e}")
        # Return empty figure on error
        return go.Figure()


def interpolate_vectors_at_temperature(vector_snapshots: Dict[float, np.ndarray], target_temp: float) -> np.ndarray:
    """
    Interpolate vectors at exactly the target temperature using the two closest snapshots.
    
    Args:
        vector_snapshots: Dictionary mapping temperature to vectors
        target_temp: Target temperature for interpolation
        
    Returns:
        Interpolated vectors at target temperature
    """
    if not vector_snapshots:
        return None
    
    available_temps = sorted(vector_snapshots.keys())
    
    # Find the two closest temperatures
    if target_temp <= available_temps[0]:
        # Target is below or at the lowest available temperature
        return vector_snapshots[available_temps[0]]
    elif target_temp >= available_temps[-1]:
        # Target is above or at the highest available temperature
        return vector_snapshots[available_temps[-1]]
    else:
        # Find the two temperatures that bracket the target
        for i in range(len(available_temps) - 1):
            if available_temps[i] <= target_temp <= available_temps[i + 1]:
                t1, t2 = available_temps[i], available_temps[i + 1]
                v1, v2 = vector_snapshots[t1], vector_snapshots[t2]
                
                # Linear interpolation weight
                alpha = (target_temp - t1) / (t2 - t1)
                
                # Interpolate vectors
                interpolated_vectors = (1 - alpha) * v1 + alpha * v2
                
                # Renormalize to unit length
                interpolated_vectors = interpolated_vectors / np.linalg.norm(interpolated_vectors, axis=1, keepdims=True)
                
                return interpolated_vectors
    
    return None


def plot_full_umap_projection(
    vectors_at_temp: np.ndarray, 
    dynamics_languages: List[str],
    analysis_results: Dict[str, Any],
    anchor_language: Optional[str] = None, 
    include_anchor: bool = False,
    target_temp: Optional[float] = None
) -> go.Figure:
    """
    Plot full UMAP projection with Ising-compatible clustering.
    Clustering is performed in spin space (768D), UMAP is only for visualization.
    """
    try:
        if vectors_at_temp is None or vectors_at_temp.shape[0] == 0:
            st.warning(f"No vector data provided for temperature {target_temp:.3f}.")
            return go.Figure()

        # Perform UMAP projection for visualization only
        from umap import UMAP
        if vectors_at_temp.shape[0] > 2:
            n_neighbors = min(15, vectors_at_temp.shape[0] - 1)
            umap_model = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
            projected_vectors = umap_model.fit_transform(vectors_at_temp)
        else:
            projected_vectors = np.random.rand(vectors_at_temp.shape[0], 2) # Fallback

        # ISING-COMPATIBLE CLUSTERING: Cluster in spin space (768D)
        from core.clustering import ising_compatible_clustering
        
        # Use temperature for adaptive thresholding
        temp_for_clustering = target_temp if target_temp is not None else 1.0
        
        # Get critical temperature from analysis results or session state
        critical_temperature = None
        if analysis_results and 'critical_temperature' in analysis_results:
            critical_temperature = analysis_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            critical_temperature = st.session_state.critical_temperature
        
        # Perform clustering in original vector space (spin space)
        clustering_result = ising_compatible_clustering(
            vectors_at_temp, 
            temperature=temp_for_clustering,
            critical_temperature=critical_temperature,
            min_cluster_size=2  # Use minimum cluster size of 2 for meaningful clusters
        )
        
        # Extract cluster labels from spin space clustering
        cluster_labels = clustering_result['cluster_labels']
        n_clusters = clustering_result['n_clusters']
        cluster_entropy = clustering_result['cluster_entropy']
        largest_cluster_size = clustering_result['largest_cluster_size']
        threshold_used = clustering_result['threshold']
        
        print(f"DEBUG: Ising clustering complete - {n_clusters} clusters, entropy: {cluster_entropy:.3f}, largest: {largest_cluster_size}, threshold: {threshold_used:.3f}")

        # Prepare data for plotting (color by spin space clusters)
        df = pd.DataFrame(projected_vectors, columns=['UMAP 1', 'UMAP 2'])
        df['language'] = dynamics_languages
        df['language_name'] = [LANGUAGE_NAMES.get(lang, lang) for lang in dynamics_languages]
        df['cluster'] = [f'Cluster {c+1}' for c in cluster_labels]

        # Create scatter plot colored by spin space clusters
        fig = px.scatter(
            df, x='UMAP 1', y='UMAP 2', color='cluster', text='language',
            hover_name='language_name',
            hover_data={'UMAP 1': ':.3f', 'UMAP 2': ':.3f', 'cluster': True, 'language': False},
            title=f"UMAP Projection at T = {target_temp:.3f} (Spin Space Clusters)"
        )
        fig.update_traces(textposition='top center', marker=dict(size=12))
        
        # Add cluster statistics to the plot
        cluster_info = f"Clusters: {n_clusters} | Entropy: {cluster_entropy:.3f} | Largest: {largest_cluster_size} | Threshold: {threshold_used:.3f}"
        fig.add_annotation(
            text=cluster_info,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.1)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Add Meta-Vector (Note: meta_vector and anchor_vector are not part of UMAP fitting)
        meta_vector = analysis_results.get('meta_vector_at_tc')
        if meta_vector is not None and 'umap_model' in locals():
            try:
                projected_meta = umap_model.transform(meta_vector.reshape(1, -1))
                fig.add_trace(go.Scatter(
                    x=projected_meta[:, 0], y=projected_meta[:, 1],
                    mode='markers+text', text=["Meta"], textposition="bottom center",
                    marker=dict(symbol='star', color='red', size=18, line=dict(width=1, color='white')),
                    name='Meta-Vector'
                ))
            except Exception as e:
                logger.warning(f"Could not transform meta-vector for plotting: {e}")

        # Add Anchor Vector - FIXED: Get from session state directly
        anchor_vector = None
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'simulation_results'):
            anchor_vector = st.session_state.simulation_results.get('anchor_vector')
        
        # Debug anchor vector retrieval
        print(f"DEBUG: anchor_vector found: {anchor_vector is not None}")
        print(f"DEBUG: include_anchor: {include_anchor}")
        print(f"DEBUG: anchor_language: {anchor_language}")
        
        if anchor_vector is not None and anchor_language and 'umap_model' in locals():
            try:
                # Handle both single vector and array of vectors
                if anchor_vector.ndim == 1:
                    anchor_vector = anchor_vector.reshape(1, -1)
                elif anchor_vector.ndim > 2:
                    anchor_vector = anchor_vector.reshape(1, -1)
                
                projected_anchor = umap_model.transform(anchor_vector)
                fig.add_trace(go.Scatter(
                    x=projected_anchor[:, 0], y=projected_anchor[:, 1],
                    mode='markers+text', text=[anchor_language.upper()], textposition="bottom center",
                    marker=dict(symbol='star', color='yellow', size=18, line=dict(width=1, color='white')),
                    name=f'Anchor ({anchor_language.upper()})'
                ))
                print(f"DEBUG: Anchor vector added successfully")
            except Exception as e:
                logger.warning(f"Could not transform anchor vector for plotting: {e}")
                print(f"DEBUG: Anchor vector transformation failed: {e}")

        fig.update_layout(template="plotly_dark", legend_title_text='Spin Space Clusters')
        return fig

    except Exception as e:
        logger.error(f"Error creating full UMAP projection: {e}", exc_info=True)
        fig = go.Figure()
        fig.update_layout(title="Error Generating UMAP Plot", template="plotly_dark")
        fig.add_annotation(text=f"An error occurred: {e}", showarrow=False)
        return fig


def plot_correlation_decay(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Plot correlation decay vs distance (log-log)
    
    Args:
        analysis_results: Dictionary containing correlation analysis data
        
    Returns:
        plotly.graph_objects.Figure: Interactive correlation decay plot
    """
    try:
        # Check if required data exists
        if 'correlation_analysis' not in analysis_results:
            logger.warning("No 'correlation_analysis' key found in analysis_results")
            return go.Figure()
        
        correlation_analysis = analysis_results['correlation_analysis']
        if 'correlation_matrix' not in correlation_analysis:
            logger.warning("No 'correlation_matrix' in correlation_analysis")
            return go.Figure()
        
        correlation_matrix = correlation_analysis['correlation_matrix']
        
        # Check if correlation matrix is valid
        if correlation_matrix.size == 0:
            logger.warning("Empty correlation matrix")
            return go.Figure()
        
        # Compute average correlation as function of distance
        n = len(correlation_matrix)
        distances = []
        correlations = []
        
        for i in range(n):
            for j in range(i+1, n):
                distances.append(j - i)  # Distance in index space
                correlations.append(abs(correlation_matrix[i, j]))
        
        if not distances:
            logger.warning("No correlation data available")
            return go.Figure()
        
        # Create log-log plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances,
            y=correlations,
            mode='markers',
            name='Correlation',
            marker=dict(
                size=6,
                color='#ff7f0e',
                opacity=0.7
            ),
            hovertemplate='Distance: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
        ))
        
        # Update axes to log scale using the correct API
        fig.update_layout(
            xaxis=dict(type="log", title="Distance"),
            yaxis=dict(type="log", title="Correlation"),
            title="Correlation Decay vs Distance",
            template="plotly_dark",
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation decay plot: {e}")
        return go.Figure()


def plot_correlation_length_vs_temperature(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot correlation length vs temperature with Tc marker
    
    Args:
        simulation_results: Dictionary containing metrics and critical temperature
        
    Returns:
        plotly.graph_objects.Figure: Interactive correlation length vs temperature plot
    """
    try:
        # Check if required data exists
        if 'temperatures' not in simulation_results or 'correlation_length' not in simulation_results:
            logger.warning("Missing 'temperatures' or 'correlation_length' in simulation_results")
            return go.Figure()
        
        temperatures = simulation_results['temperatures']
        correlation_length = simulation_results['correlation_length']
        
        # Check for valid data
        if len(temperatures) == 0 or len(correlation_length) == 0:
            logger.warning("Empty temperature or correlation length data")
            return go.Figure()
        
        # Filter out NaN values
        valid_mask = np.isfinite(temperatures) & np.isfinite(correlation_length)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) data points for correlation length plot")
            return go.Figure()
        
        valid_temps = temperatures[valid_mask]
        valid_corr = correlation_length[valid_mask]
        
        # Create the main plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_temps,
            y=valid_corr,
            mode='lines+markers',
            name='Correlation Length',
            line=dict(color='yellow', width=1.5),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            try:
                tc_formatted = f"Tc = {float(tc):.3f}"
            except (ValueError, TypeError):
                tc_formatted = f"Tc = {tc}"
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=tc_formatted,
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title="Correlation Length vs Temperature",
            xaxis_title="Temperature",
            yaxis_title="Correlation Length (ξ)",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation length plot: {e}")
        return go.Figure()


def plot_alignment_vs_temperature(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot alignment vs temperature with convergence status coloring
    
    Args:
        simulation_results: Dictionary containing metrics and convergence data
        
    Returns:
        plotly.graph_objects.Figure: Interactive alignment vs temperature plot
    """
    try:
        # Check if required data exists
        if 'temperatures' not in simulation_results or 'alignment' not in simulation_results:
            logger.warning("Missing 'temperatures' or 'alignment' in simulation_results")
            return go.Figure()
        
        temperatures = simulation_results['temperatures']
        alignment = simulation_results['alignment']
        
        # Check for valid data
        if len(temperatures) == 0 or len(alignment) == 0:
            logger.warning("Empty temperature or alignment data")
            return go.Figure()
        
        # Filter out NaN values
        valid_mask = np.isfinite(temperatures) & np.isfinite(alignment)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) data points for alignment plot")
            return go.Figure()
        
        valid_temps = temperatures[valid_mask]
        valid_align = alignment[valid_mask]
        
        # Create figure
        fig = go.Figure()
        
        # Default color
        colors = ['blue'] * len(valid_temps)
        
        # Color code based on convergence status if available
        if 'convergence_data' in simulation_results:
            convergence_data = simulation_results['convergence_data']
            # Create a mapping from temperature to status
            temp_to_status = {data['temperature']: data['status'] for data in convergence_data}
            
            # Color code based on status
            for i, temp in enumerate(valid_temps):
                status = temp_to_status.get(temp, 'unknown')
                if status == 'converged':
                    colors[i] = 'green'
                elif status == 'plateau':
                    colors[i] = 'orange'
                elif status == 'diverging':
                    colors[i] = 'red'
                elif status == 'error':
                    colors[i] = 'gray'
                else:
                    colors[i] = 'blue'
        
        # Plot alignment curve
        # Safely format text for hover
        text_array = []
        for t, a in zip(valid_temps, valid_align):
            try:
                text_array.append(f"T={float(t):.3f}<br>Alignment={float(a):.4f}")
            except (ValueError, TypeError):
                text_array.append(f"T={t}<br>Alignment={a}")
        
        fig.add_trace(go.Scatter(
            x=valid_temps,
            y=valid_align,
            mode='lines+markers',
            name='Alignment',
            line=dict(color='blue', width=1.5),
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=text_array,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            try:
                tc_formatted = f"Tc = {float(tc):.3f}"
            except (ValueError, TypeError):
                tc_formatted = f"Tc = {tc}"
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=tc_formatted,
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title="Alignment vs Temperature",
            xaxis_title="Temperature",
            yaxis_title="Alignment",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating alignment plot: {e}")
        return go.Figure()


def plot_energy_vs_temperature(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot energy vs temperature with Tc marker
    
    Args:
        simulation_results: Dictionary containing metrics and critical temperature
        
    Returns:
        plotly.graph_objects.Figure: Interactive energy vs temperature plot
    """
    try:
        # Check if required data exists
        if 'temperatures' not in simulation_results or 'energy' not in simulation_results:
            logger.warning("Missing 'temperatures' or 'energy' in simulation_results")
            return go.Figure()
        
        temperatures = simulation_results['temperatures']
        energy = simulation_results['energy']
        
        # Check for valid data
        if len(temperatures) == 0 or len(energy) == 0:
            logger.warning("Empty temperature or energy data")
            return go.Figure()
        
        # Filter out NaN values
        valid_mask = np.isfinite(temperatures) & np.isfinite(energy)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) data points for energy plot")
            return go.Figure()
        
        valid_temps = temperatures[valid_mask]
        valid_energy = energy[valid_mask]
        
        # Create the main plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_temps,
            y=valid_energy,
            mode='lines+markers',
            name='Energy',
            line=dict(color='#2ca02c', width=1.5),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            try:
                tc_formatted = f"Tc = {float(tc):.3f}"
            except (ValueError, TypeError):
                tc_formatted = f"Tc = {tc}"
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=tc_formatted,
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title="Energy vs Temperature",
            xaxis_title="Temperature",
            yaxis_title="Energy",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating energy plot: {e}")
        return go.Figure()


def plot_convergence_summary(convergence_data: List[Dict[str, Any]], tc: float = None) -> go.Figure:
    """
    Plot convergence summary across all temperatures.
    
    Args:
        convergence_data: List of convergence data from simulation
        tc: Critical temperature
        
    Returns:
        Plotly figure showing convergence status across temperatures
    """
    fig = go.Figure()
    
    # Show summary across all temperatures
    temperatures = []
    final_diffs = []
    statuses = []
    iterations = []
    
    for data in convergence_data:
        temperatures.append(data['temperature'])
        final_diffs.append(data['final_diff'])
        statuses.append(data['status'])
        iterations.append(data['iterations'])
    
    # Check for very small final_diff values that might cause log-scale issues
    min_diff = min(final_diffs)
    max_diff = max(final_diffs)
    
    # If all diffs are very small, use linear scale instead of log
    use_log_scale = True
    if max_diff < 1e-6:
        use_log_scale = False
    elif min_diff < 1e-10:
        # Replace extremely small values with a minimum threshold
        final_diffs = [max(d, 1e-10) for d in final_diffs]
    
    # Color code by status
    colors = []
    for status in statuses:
        if status == 'converged':
            colors.append('green')
        elif status == 'plateau':
            colors.append('orange')
        elif status == 'diverging':
            colors.append('red')
        elif status == 'max_steps':
            colors.append('purple')
        else:
            colors.append('gray')
    
    # Safely format text for hover
    text_array = []
    for t, s, i, d in zip(temperatures, statuses, iterations, final_diffs):
        try:
            text_array.append(f"T={float(t):.3f}<br>Status: {s}<br>Iterations: {i}<br>Final diff: {float(d):.2e}")
        except (ValueError, TypeError):
            text_array.append(f"T={t}<br>Status: {s}<br>Iterations: {i}<br>Final diff: {d}")
    
    fig.add_trace(go.Scatter(
        x=temperatures,
        y=final_diffs,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=1, color='black')
        ),
        text=text_array,
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Convergence Summary'
    ))
    
    # Remove the horizontal convergence threshold line
    # Add vertical Tc line if provided
    if tc is not None:
        try:
            tc_formatted = f"Tc = {float(tc):.3f}"
        except (ValueError, TypeError):
            tc_formatted = f"Tc = {tc}"
        fig.add_vline(x=tc, line_dash="dash", line_color="red", annotation_text=tc_formatted, annotation_position="top left")
    
    layout_kwargs = {
        "title": "Convergence Summary Across Temperatures",
        "xaxis_title": "Temperature",
        "yaxis_title": "Final Difference",
        "showlegend": False,
        "template": "plotly_dark"
    }
    
    if use_log_scale:
        layout_kwargs["yaxis_type"] = "log"
    
    fig.update_layout(**layout_kwargs)
    
    return fig


def plot_convergence_history(convergence_data: List[Dict[str, Any]], selected_temperature: float = None, simulation_results: dict = None) -> go.Figure:
    """
    Plot convergence history for temperature points.
    If simulation_results['entropy_evolution_at_tc'] is present, use it for entropy evolution at Tc.
    Args:
        convergence_data: List of convergence data from simulation
        selected_temperature: Optional specific temperature to highlight
        simulation_results: Full simulation results dict (for entropy_evolution_at_tc)
    Returns:
        Plotly figure showing convergence history
    """
    fig = go.Figure()

    # If entropy_evolution_at_tc is present and not doing a selected_temperature plot, use it
    if simulation_results is not None and selected_temperature is None and 'entropy_evolution_at_tc' in simulation_results:
        info = simulation_results['entropy_evolution_at_tc']
        tc = simulation_results.get('critical_temperature', None)
        steps = info.get('logged_steps', [])
        alignments = info.get('alignment_history', [])
        
        # Validate data before plotting
        if steps and alignments and len(steps) > 1 and len(alignments) > 1:
            entropies = [1.0 - align for align in alignments]
            fig.add_trace(go.Scatter(
                x=steps,
                y=entropies,
                mode='lines+markers',
                name=f'Entropy Evolution at Tc = {tc:.3f}' if tc is not None else 'Entropy Evolution at Tc',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            # Safe title formatting
            tc_str = f"{tc:.3f}" if tc is not None else "Unknown"
            temp_str = f"{info.get('temperature', tc):.3f}" if info.get('temperature') is not None else "Unknown"
            
            fig.update_layout(
                title=f"Entropy Evolution at Critical Temperature (Detected Tc = {tc_str}, Showing T = {temp_str})",
                xaxis_title="Iteration",
                yaxis_title="Entropy (1 - Alignment)",
                showlegend=True
            )
            return fig
        else:
            # Not enough data for meaningful plot
            fig.add_annotation(
                text="Insufficient entropy evolution data at Tc<br>(simulation converged too quickly)",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            fig.update_layout(
                title="Entropy Evolution at Critical Temperature",
                xaxis_title="Iteration",
                yaxis_title="Entropy (1 - Alignment)"
            )
            return fig
    elif simulation_results is not None and selected_temperature is None:
        # entropy_evolution_at_tc not found, fall back to old logic
        pass

    # If a specific temperature is selected, show detailed convergence for that T
    if selected_temperature is not None:
        # Find the closest temperature
        temps = [data['temperature'] for data in convergence_data]
        closest_idx = min(range(len(temps)), key=lambda i: abs(temps[i] - selected_temperature))
        data = convergence_data[closest_idx]
        
        if data['convergence_infos']:
            # Plot convergence history for the first sweep
            conv_info = data['convergence_infos'][0]
            if conv_info['alignment_history']:
                steps = conv_info['logged_steps']
                alignments = conv_info['alignment_history']
                
                # Convert alignment to entropy (entropy = 1 - alignment for normalized values)
                entropies = [1.0 - align for align in alignments]
                
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=entropies,
                    mode='lines+markers',
                    name=f'Entropy Evolution at T={data["temperature"]:.3f}',
                    line=dict(color='blue', width=1.5),
                    marker=dict(size=6)
                ))
                
                # Add critical temperature marker if this temperature is close to Tc
                tc = None
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
                    tc = st.session_state.critical_temperature
                
                if tc is not None and abs(data['temperature'] - tc) < 0.1:
                    # Find the iteration where entropy stabilizes (closest to final value)
                    final_entropy = entropies[-1]
                    stable_idx = len(entropies) - 1
                    for i, entropy in enumerate(entropies):
                        if abs(entropy - final_entropy) < 0.01:
                            stable_idx = i
                            break
                    
                    if stable_idx < len(steps):
                        fig.add_vline(
                            x=steps[stable_idx],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Tc reached at iteration {steps[stable_idx]}",
                            annotation_position="top right"
                        )
                
                fig.update_layout(
                    title=f"Entropy Evolution at T = {data['temperature']:.3f}",
                    xaxis_title="Iteration",
                    yaxis_title="Entropy (1 - Alignment)",
                    showlegend=True,
                    template="plotly_dark"
                )
            else:
                fig.add_annotation(
                    text=f"No entropy data available for T = {data['temperature']:.3f}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig.add_annotation(
                text=f"No entropy data available for T = {data['temperature']:.3f}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    else:
        # Show entropy evolution at critical temperature
        tc = None
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        
        if tc is not None:
            # Find the closest temperature to Tc
            temps = [data['temperature'] for data in convergence_data]
            closest_idx = min(range(len(temps)), key=lambda i: abs(temps[i] - tc))
            data = convergence_data[closest_idx]
            
            if data['convergence_infos']:
                conv_info = data['convergence_infos'][0]
                if conv_info['alignment_history']:
                    steps = conv_info['logged_steps']
                    alignments = conv_info['alignment_history']
                    entropies = [1.0 - align for align in alignments]
                    
                    fig.add_trace(go.Scatter(
                        x=steps,
                        y=entropies,
                        mode='lines+markers',
                        name=f'Entropy Evolution at Tc = {tc:.3f}',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Find when entropy stabilizes
                    final_entropy = entropies[-1]
                    stable_idx = len(entropies) - 1
                    for i, entropy in enumerate(entropies):
                        if abs(entropy - final_entropy) < 0.01:
                            stable_idx = i
                            break
                    
                    if stable_idx < len(steps):
                        fig.add_vline(
                            x=steps[stable_idx],
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"Stable at iteration {steps[stable_idx]}",
                            annotation_position="top right"
                        )
                    
                    fig.update_layout(
                        title=f"Entropy Evolution at Critical Temperature (Tc = {tc:.3f})",
                        xaxis_title="Iteration",
                        yaxis_title="Entropy (1 - Alignment)",
                        showlegend=True,
                        template="plotly_dark"
                    )
                else:
                    fig.add_annotation(
                        text=f"No entropy data available at Tc = {tc:.3f}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
            else:
                fig.add_annotation(
                    text=f"No convergence data available at Tc = {tc:.3f}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig.add_annotation(
                text="Critical temperature not detected - run simulation first",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    return fig 

def plot_entropy_vs_correlation_length(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot entropy vs correlation length with Tc marker
    
    Args:
        simulation_results: Dictionary containing metrics and critical temperature
        
    Returns:
        plotly.graph_objects.Figure: Interactive entropy vs correlation length plot
    """
    try:
        # Check if required data exists
        if 'correlation_length' not in simulation_results:
            logger.warning("Missing 'correlation_length' in simulation_results")
            return go.Figure()
        
        # Get entropy data, fallback to 1-alignment if entropy not available
        entropy = None
        if 'entropy' in simulation_results:
            entropy = simulation_results['entropy']
        elif 'alignment' in simulation_results:
            entropy = 1.0 - np.array(simulation_results['alignment'])
        else:
            logger.warning("Missing both 'entropy' and 'alignment' in simulation_results")
            return go.Figure()
        
        correlation_length = simulation_results['correlation_length']
        
        # Check for valid data
        if len(entropy) == 0 or len(correlation_length) == 0:
            logger.warning("Empty entropy or correlation length data")
            return go.Figure()
        
        # Filter out NaN values
        valid_mask = np.isfinite(entropy) & np.isfinite(correlation_length)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) data points for entropy vs correlation length plot")
            return go.Figure()
        
        valid_entropy = entropy[valid_mask]
        valid_corr = correlation_length[valid_mask]
        
        # Create the main plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_corr,
            y=valid_entropy,
            mode='lines+markers',
            name='Entropy vs Correlation Length',
            line=dict(color='#9467bd', width=1.5),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        
        if tc is not None and 'temperatures' in simulation_results:
            temperatures = simulation_results['temperatures']
            # Find the index closest to Tc
            tc_idx = np.argmin(np.abs(temperatures - tc))
            if tc_idx < len(valid_corr):
                fig.add_trace(go.Scatter(
                    x=[valid_corr[tc_idx]],
                    y=[valid_entropy[tc_idx]],
                    mode='markers',
                    name=f'Tc = {tc:.3f}',
                    marker=dict(size=12, color='red', symbol='star'),
                    hovertemplate=f"<b>Tc = {tc:.3f}</b><br>ξ=%{{x:.2f}}<br>Entropy=%{{y:.2f}}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title="Entropy vs Correlation Length",
            xaxis_title="Correlation Length (ξ)",
            yaxis_title="Entropy",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating entropy vs correlation length plot: {e}")
        return go.Figure()

def plot_power_law_analysis(power_law_analysis: Dict[str, Any]) -> go.Figure:
    """
    Plot power law analysis results showing cluster size distribution
    
    Args:
        power_law_analysis: Dictionary containing power law analysis results
        
    Returns:
        plotly.graph_objects.Figure: Interactive power law plot
    """
    try:
        # Check if required data exists
        if not power_law_analysis or 'cluster_sizes' not in power_law_analysis:
            logger.warning("Missing power law analysis data")
            return go.Figure()
        
        cluster_sizes = power_law_analysis.get('cluster_sizes', [])
        cluster_counts = power_law_analysis.get('cluster_counts', [])
        exponent = power_law_analysis.get('exponent', np.nan)
        r_squared = power_law_analysis.get('r_squared', 0.0)
        fitted_sizes = power_law_analysis.get('fitted_sizes', [])
        fitted_counts = power_law_analysis.get('fitted_counts', [])
        
        # Check for valid data
        if len(cluster_sizes) == 0 or len(cluster_counts) == 0:
            logger.warning("Empty cluster size or count data")
            return go.Figure()
        
        # Create the main plot
        fig = go.Figure()
        
        # Plot actual cluster size distribution
        fig.add_trace(go.Scatter(
            x=cluster_sizes,
            y=cluster_counts,
            mode='markers',
            name='Observed',
            marker=dict(
                color='#1f77b4',
                size=8,
                symbol='circle'
            ),
            hovertemplate='Size: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Plot fitted power law if available
        if len(fitted_sizes) > 0 and len(fitted_counts) > 0:
            fig.add_trace(go.Scatter(
                x=fitted_sizes,
                y=fitted_counts,
                mode='lines',
                name=f'Power Law Fit (α={exponent:.2f})',
                line=dict(
                    color='red',
                    width=2,
                    dash='dash'
                ),
                hovertemplate='Size: %{x}<br>Fitted Count: %{y}<extra></extra>'
            ))
        
        # Update layout
        title = f"Power Law Analysis (R² = {r_squared:.3f})"
        if not np.isnan(exponent):
            title += f", α = {exponent:.2f}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Cluster Size",
            yaxis_title="Number of Clusters",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified',
            xaxis_type='log',
            yaxis_type='log'
        )
        
        # Add annotation for power law interpretation
        if not np.isnan(exponent) and r_squared > 0.7:
            interpretation = "Strong power law behavior detected"
            color = "green"
        elif not np.isnan(exponent) and r_squared > 0.5:
            interpretation = "Moderate power law behavior detected"
            color = "orange"
        else:
            interpretation = "No significant power law behavior"
            color = "red"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=interpretation,
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor=color,
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating power law plot: {e}")
        # Return empty figure on error
        return go.Figure()

def plot_power_law_aggregate(simulation_results: Dict[str, Any]) -> go.Figure:
    """
    Plot log-log scatter of (cluster size, number of clusters) for all temperature steps,
    colored by temperature, with a log-log linear fit overlay.
    """
    cluster_stats = simulation_results.get('cluster_stats_per_temperature', [])
    if not cluster_stats:
        return go.Figure()
    
    # Aggregate all (s, N(s), T) points
    xs = []  # cluster sizes
    ys = []  # number of clusters of that size
    ts = []  # temperature
    for stat in cluster_stats:
        T = stat['temperature']
        sizes = stat['cluster_sizes']
        if not sizes:
            continue
        # Count number of clusters of each size
        unique, counts = np.unique(sizes, return_counts=True)
        for s, n in zip(unique, counts):
            xs.append(s)
            ys.append(n)
            ts.append(T)
    if not xs:
        return go.Figure()
    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(ts)
    
    # Log-log fit (only for points with s > 0 and n > 0)
    mask = (xs > 0) & (ys > 0)
    log_x = np.log(xs[mask])
    log_y = np.log(ys[mask])
    if len(log_x) > 1:
        slope, intercept, r_value, _, _ = linregress(log_x, log_y)
        fit_line = np.exp(intercept) * xs[mask] ** slope
    else:
        slope, intercept, r_value = np.nan, np.nan, 0.0
        fit_line = np.zeros_like(xs[mask])
    
    # Color by temperature using a colormap
    norm = Normalize(vmin=np.min(ts), vmax=np.max(ts))
    cmap = cm.get_cmap('plasma')
    colors = [f'rgb{cm.colors.to_rgb(cmap(norm(t)))}' for t in ts]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(
            color=ts,
            colorscale='Plasma',
            colorbar=dict(title='Temperature'),
            size=8,
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        name='(s, N(s))',
        hovertemplate='Cluster size: %{x}<br>Count: %{y}<br>Temp: %{marker.color:.3f}<extra></extra>'
    ))
    # Overlay log-log fit
    if len(log_x) > 1:
        fig.add_trace(go.Scatter(
            x=xs[mask],
            y=fit_line,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Log-Log Fit (slope={slope:.2f}, R²={r_value**2:.2f})',
            hoverinfo='skip'
        ))
    fig.update_layout(
        title='Power Law Distribution Across All Temperatures',
        xaxis_title='Cluster Size (log)',
        yaxis_title='Number of Clusters (log)',
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_dark',
        showlegend=True
    )
    # Add annotation for fit
    if len(log_x) > 1:
        fig.add_annotation(
            x=0.05, y=0.95, xref='paper', yref='paper',
            text=f'Fit slope: {slope:.2f}<br>R²: {r_value**2:.2f}',
            showarrow=False, font=dict(size=12, color='white'),
            bgcolor='rgba(0,0,0,0.7)'
        )
    return fig 

def plot_cluster_evolution(cluster_evolution_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Plot cluster evolution over temperature - the phase transition signature.
    
    Args:
        cluster_evolution_data: List of cluster evolution data from track_cluster_evolution()
        
    Returns:
        plotly.graph_objects.Figure: Interactive cluster evolution plot
    """
    try:
        if not cluster_evolution_data:
            logger.warning("No cluster evolution data provided")
            return go.Figure()
        
        # Extract data
        temperatures = [data['temperature'] for data in cluster_evolution_data]
        n_clusters = [data['n_clusters'] for data in cluster_evolution_data]
        cluster_entropy = [data['cluster_entropy'] for data in cluster_evolution_data]
        largest_cluster_size = [data['largest_cluster_size'] for data in cluster_evolution_data]
        thresholds = [data['threshold'] for data in cluster_evolution_data]
        
        # Create subplot figure
        fig = go.Figure()
        
        # Add traces for different metrics
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=n_clusters,
            mode='lines+markers',
            name='Number of Clusters',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=cluster_entropy,
            mode='lines+markers',
            name='Cluster Entropy',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
            yaxis='y2'
        ))
        
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=largest_cluster_size,
            mode='lines+markers',
            name='Largest Cluster Size',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6),
            yaxis='y3'
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=thresholds,
            mode='lines',
            name='Adaptive Threshold',
            line=dict(color='#d62728', width=1, dash='dash'),
            yaxis='y4'
        ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Cluster Evolution Over Temperature (Phase Transition Signature)",
            xaxis_title="Temperature",
            template="plotly_dark",
            showlegend=True,
            hovermode='x unified',
            yaxis=dict(
                title="Number of Clusters",
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
                side="left"
            ),
            yaxis2=dict(
                title="Cluster Entropy",
                titlefont=dict(color="#ff7f0e"),
                tickfont=dict(color="#ff7f0e"),
                side="right",
                overlaying="y",
                position=0.05
            ),
            yaxis3=dict(
                title="Largest Cluster Size",
                titlefont=dict(color="#2ca02c"),
                tickfont=dict(color="#2ca02c"),
                side="right",
                overlaying="y",
                position=0.15
            ),
            yaxis4=dict(
                title="Adaptive Threshold",
                titlefont=dict(color="#d62728"),
                tickfont=dict(color="#d62728"),
                side="right",
                overlaying="y",
                position=0.25
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating cluster evolution plot: {e}")
        return go.Figure() 