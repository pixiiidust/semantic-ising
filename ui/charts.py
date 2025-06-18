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
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Tc = {tc:.3f}",
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


def plot_full_umap_projection(simulation_results: Dict[str, Any], analysis_results: Dict[str, Any], 
                             anchor_language: str = None, include_anchor: bool = False) -> go.Figure:
    """
    Plot full UMAP projection of vectors at Tc (or closest available snapshot).
    If no snapshots, fallback to dynamics_vectors.
    Highlights anchor language if provided.
    """
    try:
        tc = analysis_results.get('critical_temperature')
        vector_snapshots = simulation_results.get('vector_snapshots', {})
        languages = simulation_results.get('languages', [f'Lang_{i}' for i in range(len(simulation_results.get('dynamics_vectors', [])))])
        
        tc_vectors = None
        used_temp = None
        fallback = False
        interpolation_used = False
        
        # Try to get vectors at exactly Tc using interpolation
        if vector_snapshots and len(vector_snapshots) >= 2:
            tc_vectors = interpolate_vectors_at_temperature(vector_snapshots, tc)
            if tc_vectors is not None:
                used_temp = tc
                interpolation_used = True
        
        # If interpolation failed, try closest snapshot
        if tc_vectors is None and vector_snapshots:
            available_temps = list(vector_snapshots.keys())
            if len(available_temps) > 0:
                closest_temp = min(available_temps, key=lambda t: abs(t - tc))
                tc_vectors = vector_snapshots[closest_temp]
                used_temp = closest_temp
        
        # Fallback to dynamics_vectors if no snapshots
        if tc_vectors is None:
            tc_vectors = simulation_results.get('dynamics_vectors', None)
            used_temp = tc
            fallback = True
            
        if tc_vectors is None:
            # Nothing to plot
            return go.Figure()
            
        # Perform UMAP projection
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(tc_vectors)
        except ImportError:
            return go.Figure()
            
        # Adjust languages if needed
        if len(languages) != len(tc_vectors):
            warning_msg = f"[Warning: {len(languages)} language codes, {len(tc_vectors)} vectors. Showing generic labels.]"
            languages = [f'Lang_{i}' for i in range(len(tc_vectors))]
        else:
            warning_msg = None
        
        # Prepare hover texts with language code and name
        hover_texts = [
            f"{code} = {LANGUAGE_NAMES.get(code, 'Unknown')}" for code in languages
        ]
        
        # Create scatter plot with anchor language highlighting
        fig = go.Figure()
        
        # Separate anchor language from other languages if anchor is included
        if anchor_language and include_anchor and anchor_language in languages:
            # Find anchor language index
            anchor_idx = languages.index(anchor_language)
            
            # Plot non-anchor languages first
            non_anchor_indices = [i for i in range(len(languages)) if i != anchor_idx]
            if non_anchor_indices:
                fig.add_trace(go.Scatter(
                    x=coords[non_anchor_indices, 0],
                    y=coords[non_anchor_indices, 1],
                    mode='markers+text',
                    name='Other Languages',
                    text=[languages[i] for i in non_anchor_indices],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color='#636EFA',
                        line=dict(width=1, color='white')
                    ),
                    customdata=[hover_texts[i] for i in non_anchor_indices],
                    hovertemplate='<b>%{text}</b><br>%{customdata}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
                ))
            
            # Plot anchor language with highlighting
            fig.add_trace(go.Scatter(
                x=[coords[anchor_idx, 0]],
                y=[coords[anchor_idx, 1]],
                mode='markers+text',
                name=f'Anchor ({anchor_language})',
                text=[anchor_language],
                textposition="top center",
                marker=dict(
                    size=15,
                    color='#FF6B6B',
                    line=dict(width=2, color='white'),
                    symbol='star'
                ),
                customdata=[hover_texts[anchor_idx]],
                hovertemplate=f'<b>{anchor_language} (Anchor)</b><br>%{{customdata}}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>'
            ))
        else:
            # Plot all languages normally (no anchor highlighting)
            fig.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers+text',
                name='Languages',
                text=languages,
                textposition="top center",
                marker=dict(
                    size=10,
                    color='#636EFA',
                    line=dict(width=1, color='white')
                ),
                customdata=hover_texts,
                hovertemplate='<b>%{text}</b><br>%{customdata}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
        
        # Update layout with appropriate title
        if interpolation_used:
            title = f"UMAP Projection at T = {used_temp:.3f} (interpolated)"
        elif fallback:
            title = f"UMAP Projection at T = {used_temp:.3f} (original vectors, no snapshots)"
        else:
            title = f"UMAP Projection at T = {used_temp:.3f} (closest snapshot)"
        
        # Add anchor language info to title if applicable
        if anchor_language and include_anchor:
            title += f" - {anchor_language} highlighted"
        if warning_msg:
            title += f"<br>{warning_msg}"
        
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            template="plotly_dark",
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating UMAP projection: {e}")
        return go.Figure()


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
            line=dict(color='#d62728', width=2),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Tc = {tc:.3f}",
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
        fig.add_trace(go.Scatter(
            x=valid_temps,
            y=valid_align,
            mode='lines+markers',
            name='Alignment',
            line=dict(color='blue', width=2),
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=[f"T={t:.3f}<br>Alignment={a:.4f}" for t, a in zip(valid_temps, valid_align)],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Tc = {tc:.3f}",
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
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6)
        ))
        
        # Add critical temperature marker if available
        tc = None
        if 'critical_temperature' in simulation_results:
            tc = simulation_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            tc = st.session_state.critical_temperature
        if tc is not None:
            fig.add_vline(
                x=tc,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Tc = {tc:.3f}",
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


def plot_convergence_history(convergence_data: List[Dict[str, Any]], selected_temperature: float = None) -> go.Figure:
    """
    Plot convergence history for temperature points.
    
    Args:
        convergence_data: List of convergence data from simulation
        selected_temperature: Optional specific temperature to highlight
        
    Returns:
        Plotly figure showing convergence history
    """
    fig = go.Figure()
    
    # If a specific temperature is selected, show detailed convergence for that T
    if selected_temperature is not None:
        # Find the closest temperature
        temps = [data['temperature'] for data in convergence_data]
        closest_idx = min(range(len(temps)), key=lambda i: abs(temps[i] - selected_temperature))
        data = convergence_data[closest_idx]
        
        if data['convergence_infos']:
            # Plot convergence history for the first sweep
            conv_info = data['convergence_infos'][0]
            if conv_info['diff_history']:
                steps = conv_info['logged_steps']
                diffs = conv_info['diff_history']
                
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=diffs,
                    mode='lines+markers',
                    name=f'Convergence at T={data["temperature"]:.3f}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                # Add convergence threshold line
                fig.add_hline(
                    y=1e-3,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Convergence Threshold (1e-3)",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title=f"Convergence History at T = {data['temperature']:.3f}",
                    xaxis_title="Iteration",
                    yaxis_title="Difference (||v_new - v_old||/||v_old||)",
                    yaxis_type="log",
                    showlegend=True
                )
            else:
                fig.add_annotation(
                    text=f"No convergence data available for T = {data['temperature']:.3f}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            fig.add_annotation(
                text=f"No convergence data available for T = {data['temperature']:.3f}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    else:
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
        
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=final_diffs,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=[f"T={t:.3f}<br>Status: {s}<br>Iterations: {i}<br>Final diff: {d:.2e}" 
                  for t, s, i, d in zip(temperatures, statuses, iterations, final_diffs)],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Convergence Summary'
        ))
        
        # Add convergence threshold line
        fig.add_hline(
            y=1e-3,
            line_dash="dash",
            line_color="red",
            annotation_text="Convergence Threshold",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Convergence Summary Across Temperatures",
            xaxis_title="Temperature",
            yaxis_title="Final Difference",
            yaxis_type="log",
            showlegend=False
        )
    
    return fig 