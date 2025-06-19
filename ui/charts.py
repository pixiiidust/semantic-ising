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
    Highlights anchor language and meta-vector if provided.
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
        
        # Compute meta-vector from multilingual set
        from core.meta_vector import compute_meta_vector
        meta_result = compute_meta_vector(tc_vectors, method="centroid")
        meta_vector = meta_result['meta_vector']
        
        # Prepare vectors for UMAP projection
        anchor_vector = None  # Initialize anchor_vector
        
        if anchor_language and not include_anchor:
            # Get anchor vector from the original embeddings
            from core.embeddings import generate_embeddings
            try:
                # Get all embeddings including anchor using the same parameters as simulation
                concept = simulation_results.get('concept', 'dog')
                encoder = simulation_results.get('encoder', 'LaBSE')
                
                # Try to get the filename from session state if available
                filename = None
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'concept_info'):
                    filename = st.session_state.concept_info.get('filename')
                
                logger.info(f"Retrieving anchor vector for {anchor_language} from {concept} with {encoder}, filename: {filename}")
                
                all_embeddings, all_languages = generate_embeddings(concept, encoder, filename)
                anchor_idx = all_languages.index(anchor_language)
                anchor_vector = all_embeddings[anchor_idx:anchor_idx+1]  # Keep 2D shape
                
                logger.info(f"Successfully retrieved anchor vector for {anchor_language} at index {anchor_idx}")
                
                # Combine vectors for UMAP: dynamics vectors + meta-vector + anchor vector
                all_vectors = np.vstack([tc_vectors, meta_vector.reshape(1, -1), anchor_vector])
            except Exception as e:
                logger.warning(f"Failed to retrieve anchor vector for {anchor_language}: {e}")
                # Fallback: just use dynamics vectors + meta-vector
                all_vectors = np.vstack([tc_vectors, meta_vector.reshape(1, -1)])
                anchor_vector = None
        else:
            # Combine vectors for UMAP projection: language vectors + meta-vector
            all_vectors = np.vstack([tc_vectors, meta_vector.reshape(1, -1)])
            # Note: anchor_vector remains None for included case (will be extracted from dynamics later)
        
        # Perform UMAP projection on all vectors
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(all_vectors)
        except ImportError:
            return go.Figure()
        
        # Separate coordinates for language vectors, meta-vector, and anchor vector (if present)
        if anchor_vector is not None:
            # Format: [language_vectors, meta_vector, anchor_vector]
            language_coords = coords[:-2]  # All except last two
            meta_coords = coords[-2]       # Second to last is meta-vector
            anchor_coords = coords[-1]     # Last one is anchor vector
            logger.info(f"UMAP projection: {len(language_coords)} language vectors + meta-vector + anchor vector")
        else:
            # Format: [language_vectors, meta_vector]
            language_coords = coords[:-1]  # All except last
            meta_coords = coords[-1]       # Last one is meta-vector
            anchor_coords = None
            logger.info(f"UMAP projection: {len(language_coords)} language vectors + meta-vector (no anchor)")
        
        # If anchor is included in dynamics, find its position and highlight it
        if include_anchor and anchor_language and anchor_coords is None:
            try:
                # Find the anchor language index in the dynamics languages
                anchor_idx = languages.index(anchor_language)
                if anchor_idx < len(language_coords):
                    # Extract anchor coordinates from language coordinates
                    anchor_coords = language_coords[anchor_idx]
                    # Remove anchor from language coordinates to avoid duplication
                    language_coords = np.vstack([language_coords[:anchor_idx], language_coords[anchor_idx+1:]])
                    # Remove anchor from languages list
                    languages = languages[:anchor_idx] + languages[anchor_idx+1:]
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not find anchor language {anchor_language} in dynamics: {e}")
                anchor_coords = None
        
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
        
        # Create scatter plot with anchor language and meta-vector highlighting
        fig = go.Figure()
        
        # Plot all language vectors first (these are the dynamics languages)
        fig.add_trace(go.Scatter(
            x=language_coords[:, 0],
            y=language_coords[:, 1],
            mode='markers+text',
            name='Multilingual Set',
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
        
        # Plot meta-vector as red circle
        fig.add_trace(go.Scatter(
            x=[meta_coords[0]],
            y=[meta_coords[1]],
            mode='markers+text',
            name='Meta-Vector',
            text=['Meta'],
            textposition="top center",
            marker=dict(
                size=15,
                color='#FF6B6B',
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            customdata=['Meta-Vector (centroid of multilingual set)'],
            hovertemplate='<b>Meta-Vector</b><br>Centroid of multilingual set<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ))
        
        # Plot anchor vector if present (excluded case)
        if anchor_coords is not None:
            # Determine if anchor is included or excluded
            if include_anchor:
                anchor_status = "included in multilingual set"
                anchor_title = f"{anchor_language} (Anchor - Included)"
            else:
                anchor_status = "excluded from multilingual set"
                anchor_title = f"{anchor_language} (Anchor - Excluded)"
            
            logger.info(f"Plotting anchor vector for {anchor_language}: {anchor_status} at coordinates {anchor_coords}")
                
            fig.add_trace(go.Scatter(
                x=[anchor_coords[0]],
                y=[anchor_coords[1]],
                mode='markers+text',
                name=f'Anchor ({anchor_language})',
                text=[anchor_language],
                textposition="top center",
                marker=dict(
                    size=15,
                    color='#00D4AA',  # Different color to distinguish from meta-vector
                    line=dict(width=2, color='white'),
                    symbol='diamond'  # Different symbol to distinguish from meta-vector
                ),
                customdata=[f'{anchor_language} = {LANGUAGE_NAMES.get(anchor_language, "Unknown")} ({anchor_status})'],
                hovertemplate=f'<b>{anchor_title}</b><br>{anchor_status}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>'
            ))
        else:
            logger.info(f"No anchor coordinates available for {anchor_language} (include_anchor={include_anchor})")
        
        # Update layout with appropriate title
        if interpolation_used:
            title = f"UMAP Projection at T = {used_temp:.3f} (interpolated)"
        elif fallback:
            title = f"UMAP Projection at T = {used_temp:.3f} (original vectors, no snapshots)"
        else:
            title = f"UMAP Projection at T = {used_temp:.3f} (closest snapshot)"
        
        # Add anchor language info to title
        if anchor_language:
            if include_anchor:
                title += f" - {anchor_language} included in multilingual set"
            else:
                title += f" - {anchor_language} excluded from multilingual set (shown separately)"
        if warning_msg:
            title += f"<br>{warning_msg}"
        
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            template="plotly_dark",
            showlegend=True,
            hovermode='closest',
            height=600,  # Make chart taller
            width=800,   # Set reasonable width
            margin=dict(l=50, r=50, t=80, b=50)  # Adjust margins for better proportions
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
            line=dict(color='blue', width=1.5),
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
    
    # Remove the horizontal convergence threshold line
    # Add vertical Tc line if provided
    if tc is not None:
        fig.add_vline(x=tc, line_dash="dash", line_color="red", annotation_text=f"Tc = {tc:.3f}", annotation_position="top left")
    
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