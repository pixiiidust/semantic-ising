"""
UI component functions for Streamlit interface (Phase 9)
Provides reusable UI components for anchor configuration and experiment display
"""

import streamlit as st
from typing import List, Tuple, Dict, Any
import logging
import numpy as np
import os
import json

logger = logging.getLogger(__name__)


def get_available_concepts() -> List[Dict[str, any]]:
    """
    Scan data/concepts/ folder and return available concept files
    
    Returns:
        List of dictionaries with concept info, sorted by modification time (newest first)
    """
    concepts_dir = "data/concepts"
    concept_files = []
    
    try:
        if not os.path.exists(concepts_dir):
            logger.warning(f"Concepts directory not found: {concepts_dir}")
            return []
        
        # Find all JSON files that contain translations
        for filename in os.listdir(concepts_dir):
            if filename.endswith(".json") and "translation" in filename.lower():
                filepath = os.path.join(concepts_dir, filename)
                
                # Get file modification time
                mtime = os.path.getmtime(filepath)
                
                # Extract concept name from filename
                # Handle different naming patterns:
                # - dog_translations.json -> dog
                # - dog_translations_25.json -> dog (25 languages)
                # - dog_translations_75.json -> dog (75 languages)
                # - i_love_you_translations_25.json -> i love you (25 languages)
                concept_name = filename.replace("_translations.json", "")
                concept_name = concept_name.replace("_translations_25.json", "")
                concept_name = concept_name.replace("_translations_75.json", "")
                concept_name = concept_name.replace("_translations_72.json", "")  # Legacy support
                concept_name = concept_name.replace("_translations_", "_")  # Fallback
                
                # Convert underscores to spaces for better display
                display_concept_name = concept_name.replace("_", " ")
                
                # Count languages in the file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        translations = json.load(f)
                    language_count = len(translations)
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    language_count = 0
                
                # Create display name that includes language count
                if "_72" in filename:
                    display_name = f"{display_concept_name} ({language_count} languages)"
                else:
                    display_name = f"{display_concept_name} ({language_count} languages)"
                
                concept_files.append({
                    'concept_name': concept_name,
                    'display_concept_name': display_concept_name,
                    'display_name': display_name,
                    'filename': filename,
                    'filepath': filepath,
                    'language_count': language_count,
                    'modification_time': mtime
                })
        
        # Sort by modification time (newest first)
        concept_files.sort(key=lambda x: x['modification_time'], reverse=True)
        
        return concept_files
        
    except Exception as e:
        logger.error(f"Error scanning concepts directory: {e}")
        return []


def render_concept_selection() -> Tuple[str, Dict[str, any]]:
    """
    Render dynamic concept selection with file information
    
    Returns:
        Tuple of (selected_concept_name, selected_concept_info)
    """
    # Get available concepts
    available_concepts = get_available_concepts()
    
    if not available_concepts:
        st.error("No concept files found in data/concepts/ folder")
        st.info("Please add translation JSON files to the data/concepts/ folder")
        return None, None
    
    # Create concept options for dropdown using display names
    concept_options = [concept['display_name'] for concept in available_concepts]
    
    # Concept selection dropdown
    selected_display_name = st.selectbox(
        "Concept to Simulate",
        concept_options,
        help="Select a concept to analyze across languages"
    )
    
    # Find the selected concept info
    selected_concept_info = next(
        (concept for concept in available_concepts if concept['display_name'] == selected_display_name),
        None
    )
    
    # File information display (collapsed by default)
    if selected_concept_info:
        with st.expander("📁 File Information", expanded=False):
            st.markdown(f"**Concept:** {selected_concept_info['display_concept_name'].title()}")
            st.markdown(f"**Languages:** {selected_concept_info['language_count']}")
            st.markdown(f"**File:** {selected_concept_info['filename']}")
            st.markdown(f"**Location:** {selected_concept_info['filepath']}")
            import datetime
            mod_time = datetime.datetime.fromtimestamp(selected_concept_info['modification_time'])
            st.markdown(f"**Last Modified:** {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return selected_concept_info['display_concept_name'] if selected_concept_info else None, selected_concept_info


def render_anchor_config() -> Tuple[str, bool]:
    """
    Render anchor language configuration sidebar
    
    Returns:
        Tuple[str, bool]: (anchor_language, include_anchor)
    """
    try:
        st.sidebar.subheader("🔗 Anchor Language Configuration")
        
        # Anchor language selection
        anchor_language = st.selectbox(
            "Anchor Language",
            ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
            index=0,
            help="Select the reference language for comparison"
        )
        
        # Include anchor in dynamics toggle
        include_anchor = st.checkbox(
            "Include anchor in multilingual set",
            value=False,
            help="✅ Include: Anchor participates in simulation\n❌ Exclude: Anchor compared to multilingual result"
        )
        
        # Display experiment type
        experiment_type = "Single-phase" if include_anchor else "Two-phase"
        st.info(f"**Experiment Type:** {experiment_type}")
        
        return anchor_language, include_anchor
        
    except Exception as e:
        logger.error(f"Error rendering anchor config: {e}")
        # Return defaults on error
        return "en", False


def render_experiment_description(anchor_language: str, 
                                 include_anchor: bool, 
                                 dynamics_languages: List[str]) -> None:
    """
    Display experiment configuration details
    
    Args:
        anchor_language: Selected anchor language
        include_anchor: Whether anchor is included in dynamics
        dynamics_languages: List of languages participating in dynamics
    """
    try:
        # Determine experiment type and description
        if include_anchor:
            experiment_type = "Single-phase"
            question = "Does anchor share semantic space with other languages?"
            mode_desc = f"{anchor_language} participates in Ising dynamics with {len(dynamics_languages)} languages"
        else:
            experiment_type = "Two-phase"
            question = "How does anchor compare to multilingual semantic dynamics?"
            mode_desc = f"{anchor_language} compared to Ising dynamics of {len(dynamics_languages)} languages"
        
        # Create description text
        description = f"""
        **🔬 Experiment Configuration:**
        
        - **Anchor Language:** {anchor_language}
        - **Dynamics Languages:** {len(dynamics_languages)} languages
        - **Mode:** {experiment_type} ({'anchor included' if include_anchor else 'anchor excluded'})
        - **Question:** {question}
        - **Description:** {mode_desc}
        """
        
        st.info(description)
        
    except Exception as e:
        logger.error(f"Error rendering experiment description: {e}")
        st.error("Error displaying experiment configuration")


def render_simulation_progress(progress: float, status: str = "") -> None:
    """
    Render simulation progress bar
    
    Args:
        progress: Progress value between 0 and 1
        status: Status message to display
    """
    try:
        st.progress(progress)
        if status:
            st.write(f"**Status:** {status}")
    except Exception as e:
        logger.error(f"Error rendering progress: {e}")


def render_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Render metrics summary in a clean format
    
    Args:
        metrics: Dictionary of metric names and values
    """
    try:
        st.subheader("📊 Metrics Summary")
        
        # Create columns for metrics
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=f"{value:.4f}" if isinstance(value, float) else str(value)
                )
                
    except Exception as e:
        logger.error(f"Error rendering metrics summary: {e}")


def render_critical_temperature_display(tc: float, method: str = "Binder Cumulant") -> None:
    """
    Render critical temperature display with method information
    
    Args:
        tc: Critical temperature value
        method: Method used to detect critical temperature
    """
    try:
        st.subheader("🌡️ Critical Temperature Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Critical Temperature (Tc)",
                value=f"{tc:.3f}",
                help="Temperature at which phase transition occurs"
            )
        
        with col2:
            st.metric(
                label="Detection Method",
                value=method,
                help="Algorithm used to detect critical temperature"
            )
            
    except Exception as e:
        logger.error(f"Error rendering critical temperature display: {e}")


def render_anchor_comparison_summary(comparison_metrics: Dict[str, float]) -> None:
    """
    Render anchor comparison metrics summary
    
    Args:
        comparison_metrics: Dictionary of comparison metric names and values
    """
    try:
        st.subheader("🔗 Anchor Comparison Summary")
        
        # Define metric descriptions
        metric_descriptions = {
            'cosine_distance': 'Distance between anchor language vector and multilingual meta-vector at Tc (0-1, lower is better)',
            'cosine_similarity': 'Similarity between anchor language vector and multilingual meta-vector at Tc (0-1, higher is better)'
        }
        
        # Create columns for metrics
        cols = st.columns(len(comparison_metrics))
        
        for i, (metric_name, value) in enumerate(comparison_metrics.items()):
            with cols[i]:
                description = metric_descriptions.get(metric_name, "")
                st.metric(
                    label=metric_name.replace('_', ' ').title(),
                    value=f"{value:.4f}" if isinstance(value, float) else str(value),
                    help=description
                )
                
    except Exception as e:
        logger.error(f"Error rendering anchor comparison summary: {e}")


def render_power_law_summary(power_law_analysis: Dict[str, Any]) -> None:
    """
    Render power law analysis summary
    
    Args:
        power_law_analysis: Dictionary containing power law analysis results
    """
    try:
        st.subheader("📈 Power Law Analysis")
        
        # Extract values
        exponent = power_law_analysis.get('exponent', np.nan)
        r_squared = power_law_analysis.get('r_squared', 0.0)
        n_clusters = power_law_analysis.get('n_clusters', 0)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Power Law Exponent",
                value=f"{exponent:.3f}" if not np.isnan(exponent) else "N/A",
                help="Exponent of power law distribution"
            )
        
        with col2:
            st.metric(
                label="R² Goodness of Fit",
                value=f"{r_squared:.3f}",
                help="Goodness of fit for power law (0-1 scale)"
            )
        
        with col3:
            st.metric(
                label="Number of Clusters",
                value=str(n_clusters),
                help="Total number of clusters detected"
            )
            
    except Exception as e:
        logger.error(f"Error rendering power law summary: {e}")


def render_export_buttons(simulation_results: Dict[str, Any], 
                         analysis_results: Dict[str, Any]) -> None:
    """
    Render export buttons for different data formats
    
    Args:
        simulation_results: Simulation results data
        analysis_results: Analysis results data
    """
    try:
        st.subheader("💾 Export Results")
        
        # Create columns for export buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 CSV Export", help="Export simulation results as CSV"):
                try:
                    from export.ui_helpers import export_csv_results
                    file_path = export_csv_results(simulation_results, analysis_results)
                    st.success(f"CSV exported to: {file_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col2:
            if st.button("🔢 Vectors @ Tc", help="Export vectors at critical temperature"):
                try:
                    from export.ui_helpers import export_vectors_at_tc
                    file_path = export_vectors_at_tc(simulation_results)
                    st.success(f"Vectors exported to: {file_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col3:
            if st.button("📊 Charts", help="Export charts as PNG files"):
                try:
                    from export.ui_helpers import export_charts
                    file_path = export_charts(simulation_results, analysis_results)
                    st.success(f"Charts exported to: {file_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col4:
            if st.button("⚙️ Config", help="Export current configuration as YAML"):
                try:
                    from export.ui_helpers import export_config_file
                    file_path = export_config_file()
                    st.success(f"Config exported to: {file_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
                    
    except Exception as e:
        logger.error(f"Error rendering export buttons: {e}")


def render_error_message(error: Exception, context: str = "") -> None:
    """
    Render error message in a user-friendly format
    
    Args:
        error: Exception that occurred
        context: Context where the error occurred
    """
    try:
        st.error(f"""
        **❌ Error {f'in {context}' if context else ''}:**
        
        {str(error)}
        
        Please check your configuration and try again.
        """)
    except Exception as e:
        logger.error(f"Error rendering error message: {e}")


def render_success_message(message: str) -> None:
    """
    Render success message
    
    Args:
        message: Success message to display
    """
    try:
        st.success(f"✅ {message}")
    except Exception as e:
        logger.error(f"Error rendering success message: {e}")


def render_warning_message(message: str) -> None:
    """
    Render warning message
    
    Args:
        message: Warning message to display
    """
    try:
        st.warning(f"⚠️ {message}")
    except Exception as e:
        logger.error(f"Error rendering warning message: {e}") 