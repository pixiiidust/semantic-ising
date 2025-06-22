"""
Anchor comparison tab for Streamlit interface (Phase 9)
Provides detailed analysis of anchor language comparison results
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import os
import pickle

# Import UI components
from ui.components import (
    render_anchor_comparison_summary,
    render_error_message,
    render_warning_message,
    render_success_message
)

# Import chart functions
from ui.charts import plot_full_umap_projection

# Import core simulation functions for loading
from core.simulation import _load_snapshot_from_disk, _get_available_snapshot_temperatures

logger = logging.getLogger(__name__)


def render_anchor_comparison_tab(comparison_metrics: Dict[str, float], experiment_config: Dict[str, Any]):
    """
    Render anchor comparison tab
    """
    try:
        print("=" * 50)
        print("DEBUG: render_anchor_comparison_tab CALLED")
        print("=" * 50)
        print("DEBUG: Starting render_anchor_comparison_tab")
        print(f"DEBUG: comparison_metrics keys: {list(comparison_metrics.keys()) if comparison_metrics else 'None'}")
        print(f"DEBUG: comparison_metrics values: {comparison_metrics}")
        print(f"DEBUG: experiment_config keys: {list(experiment_config.keys()) if experiment_config else 'None'}")
        
        st.header("🔗 Anchor Language Comparison")
        
        if not comparison_metrics:
            st.warning("No comparison metrics available. Run simulation first.")
            return
        
        # TEMPORARY FIX: Recalculate comparison metrics once at the beginning
        print("DEBUG: About to check for recalculation conditions")
        print(f"DEBUG: hasattr(st.session_state, 'simulation_results'): {hasattr(st.session_state, 'simulation_results')}")
        print(f"DEBUG: hasattr(st.session_state, 'analysis_results'): {hasattr(st.session_state, 'analysis_results')}")
        
        if hasattr(st.session_state, 'simulation_results') and hasattr(st.session_state, 'analysis_results'):
            print("DEBUG: Recalculation conditions met - starting recalculation")
            print("DEBUG: Recalculating comparison metrics at beginning")
            try:
                from core.comparison_metrics import compute_meta_vector
                from sklearn.metrics.pairwise import cosine_similarity
                
                simulation_results = st.session_state.simulation_results
                analysis_results = st.session_state.analysis_results
                tc = analysis_results.get('critical_temperature')
                print(f"DEBUG: Recalculation - tc: {tc}")
                
                # Check for in-memory vector_snapshots first
                if tc and 'vector_snapshots' in simulation_results and simulation_results['vector_snapshots']:
                    print("DEBUG: tc and vector_snapshots available - proceeding with calculation")
                    # Find closest temperature to Tc
                    available_temps = list(simulation_results['vector_snapshots'].keys())
                    closest_tc = min(available_temps, key=lambda t: abs(t - tc))
                    vectors_at_tc = simulation_results['vector_snapshots'][closest_tc]
                    print(f"DEBUG: Recalculation - using vectors at T={closest_tc}")
                    
                    # Calculate meta vector
                    meta_result = compute_meta_vector(vectors_at_tc)
                    meta_vector = meta_result['meta_vector']
                    
                    # Get anchor vector
                    anchor_vector = None
                    if hasattr(st.session_state, 'anchor_vectors') and st.session_state.anchor_vectors is not None:
                        anchor_vector = st.session_state.anchor_vectors[0]
                    elif 'anchor_vector' in simulation_results:
                        anchor_vector = simulation_results['anchor_vector'][0]
                    
                    if anchor_vector is not None:
                        print("DEBUG: Anchor vector found - calculating similarity")
                        # Calculate similarity
                        recalculated_similarity = cosine_similarity([anchor_vector], [meta_vector])[0][0]
                        recalculated_distance = 1 - recalculated_similarity
                        
                        print(f"DEBUG: RECALCULATED - Similarity: {recalculated_similarity:.4f}, Distance: {recalculated_distance:.4f}")
                        
                        # Update comparison_metrics with recalculated values
                        comparison_metrics = {
                            'cosine_similarity': recalculated_similarity,
                            'cosine_distance': recalculated_distance,
                            'cos_anchor_meta_vector': recalculated_similarity,
                            'avg_cos_anchor_knn': recalculated_similarity  # Placeholder
                        }
                        
                        print(f"DEBUG: Updated comparison_metrics: {comparison_metrics}")
                        
                        # UPDATE SESSION STATE so all UI components use the correct values
                        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                            st.session_state.analysis_results['anchor_comparison'] = comparison_metrics
                            print(f"DEBUG: Updated session_state.analysis_results['anchor_comparison'] with recalculated values")
                        else:
                            print(f"DEBUG: No analysis_results in session_state to update")
                    else:
                        print("DEBUG: No anchor vector found")
                
                # Check for disk-based snapshots if in-memory not available
                elif tc and 'snapshot_directory' in simulation_results and simulation_results['snapshot_directory']:
                    print("DEBUG: tc and snapshot_directory available - loading from disk")
                    snapshot_dir = simulation_results['snapshot_directory']
                    available_temps = simulation_results.get('available_snapshot_temperatures', [])
                    
                    if available_temps:
                        # Find closest temperature to Tc
                        closest_tc = min(available_temps, key=lambda t: abs(t - tc))
                        print(f"DEBUG: Recalculation - loading vectors at T={closest_tc} from disk")
                        
                        # Load snapshot from disk
                        snapshot_data = _load_snapshot_from_disk(snapshot_dir, closest_tc)
                        if snapshot_data and 'vectors' in snapshot_data:
                            vectors_at_tc = snapshot_data['vectors']
                            print(f"DEBUG: Recalculation - loaded vectors shape: {vectors_at_tc.shape}")
                            
                            # Calculate meta vector
                            meta_result = compute_meta_vector(vectors_at_tc)
                            meta_vector = meta_result['meta_vector']
                            
                            # Get anchor vector
                            anchor_vector = None
                            if hasattr(st.session_state, 'anchor_vectors') and st.session_state.anchor_vectors is not None:
                                anchor_vector = st.session_state.anchor_vectors[0]
                            elif 'anchor_vector' in simulation_results:
                                anchor_vector = simulation_results['anchor_vector'][0]
                            
                            if anchor_vector is not None:
                                print("DEBUG: Anchor vector found - calculating similarity")
                                # Calculate similarity
                                recalculated_similarity = cosine_similarity([anchor_vector], [meta_vector])[0][0]
                                recalculated_distance = 1 - recalculated_similarity
                                
                                print(f"DEBUG: RECALCULATED (DISK) - Similarity: {recalculated_similarity:.4f}, Distance: {recalculated_distance:.4f}")
                                
                                # Update comparison_metrics with recalculated values
                                comparison_metrics = {
                                    'cosine_similarity': recalculated_similarity,
                                    'cosine_distance': recalculated_distance,
                                    'cos_anchor_meta_vector': recalculated_similarity,
                                    'avg_cos_anchor_knn': recalculated_similarity  # Placeholder
                                }
                                
                                print(f"DEBUG: Updated comparison_metrics: {comparison_metrics}")
                                
                                # UPDATE SESSION STATE so all UI components use the correct values
                                if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                                    st.session_state.analysis_results['anchor_comparison'] = comparison_metrics
                                    print(f"DEBUG: Updated session_state.analysis_results['anchor_comparison'] with recalculated values")
                                else:
                                    print(f"DEBUG: No analysis_results in session_state to update")
                            else:
                                print("DEBUG: No anchor vector found")
                        else:
                            print(f"DEBUG: Failed to load snapshot data from disk for T={closest_tc}")
                    else:
                        print(f"DEBUG: No available snapshot temperatures found")
                else:
                    print(f"DEBUG: tc or snapshots not available - tc: {tc}, has vector_snapshots: {'vector_snapshots' in simulation_results}, has snapshot_directory: {'snapshot_directory' in simulation_results}")
            except Exception as e:
                print(f"DEBUG: Recalculation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("DEBUG: Recalculation conditions not met")
            print(f"DEBUG: simulation_results in session_state: {hasattr(st.session_state, 'simulation_results')}")
            print(f"DEBUG: analysis_results in session_state: {hasattr(st.session_state, 'analysis_results')}")
        
        print("DEBUG: About to display metrics in columns")
        
        # Display Critical Temperature and main metrics in 3-column layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            print("DEBUG: Processing critical temperature")
            if hasattr(st.session_state, 'critical_temperature') and st.session_state.critical_temperature:
                tc = st.session_state.critical_temperature
                print(f"DEBUG: tc value: {tc}, type: {type(tc)}")
                try:
                    tc_formatted = f"{float(tc):.3f}"
                except (ValueError, TypeError):
                    tc_formatted = f"{tc}"
                st.metric("Critical Temperature", tc_formatted, help="Critical temperature, Tc, an estimate of where phase transition occurs")
            else:
                st.metric("Critical Temperature", "N/A", help="Critical temperature not available")
        
        with col2:
            print("DEBUG: Processing cosine distance")
            # Safe formatting for cosine distance
            cosine_distance = comparison_metrics.get('cosine_distance', 0.0)
            print(f"DEBUG: cosine_distance value: {cosine_distance}, type: {type(cosine_distance)}")
            print(f"DEBUG: Full comparison_metrics: {comparison_metrics}")
            print(f"DEBUG: === COL2 DISPLAY VALUES ===")
            print(f"DEBUG: Raw cosine_distance from comparison_metrics: {comparison_metrics.get('cosine_distance')}")
            print(f"DEBUG: Raw cosine_similarity from comparison_metrics: {comparison_metrics.get('cosine_similarity')}")
            
            if cosine_distance is not None:
                try:
                    cosine_distance_formatted = f"{float(cosine_distance):.4f}"
                except (ValueError, TypeError):
                    cosine_distance_formatted = f"{cosine_distance}"
            else:
                cosine_distance_formatted = "0.0000"
            print(f"DEBUG: FINAL cosine_distance for display: {cosine_distance} -> {cosine_distance_formatted}")
            st.metric("Cosine Distance", cosine_distance_formatted, help="Distance between anchor language vector and multilingual meta-vector at Tc (0-1, lower is better)")
        
        with col3:
            print("DEBUG: Processing cosine similarity")
            # Safe formatting for cosine similarity
            cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
            print(f"DEBUG: cosine_similarity value: {cosine_similarity}, type: {type(cosine_similarity)}")
            print(f"DEBUG: === COL3 DISPLAY VALUES ===")
            print(f"DEBUG: Raw cosine_similarity from comparison_metrics: {comparison_metrics.get('cosine_similarity')}")
            
            if cosine_similarity is not None:
                try:
                    cosine_similarity_formatted = f"{float(cosine_similarity):.4f}"
                except (ValueError, TypeError):
                    cosine_similarity_formatted = f"{cosine_similarity}"
            else:
                cosine_similarity_formatted = "0.0000"
            print(f"DEBUG: FINAL cosine_similarity for display: {cosine_similarity} -> {cosine_similarity_formatted}")
            st.metric("Cosine Similarity", cosine_similarity_formatted, help="Similarity between anchor language vector and multilingual meta-vector at Tc (0-1, higher is better)")
        
        print("DEBUG: Metrics display completed")
        
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
        
        print("DEBUG: About to process interpretation")
        
        # Interpretation in expander
        with st.expander("💡 Interpretation", expanded=True):
            # Safely get cosine similarity for interpretation
            cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
            try:
                cosine_similarity_float = float(cosine_similarity) if cosine_similarity is not None else 0.0
            except (ValueError, TypeError):
                cosine_similarity_float = 0.0
            
            if cosine_similarity_float > 0.8:
                st.success("Strong semantic similarity detected")
            elif cosine_similarity_float > 0.6:
                st.warning("Moderate semantic similarity")
            else:
                st.error("Weak semantic similarity")
            
            st.write("**What's being compared:**")
            st.write("""
            * Cosine similarity and distance quantify how closely the anchor language aligns with the emergent multilingual structure at Tc."
            * The emergent multilingual structure is estimated by a **meta-vector** (centroid) computed as the average of all language vectors at the critical temperature.
            * Computation: **Meta-vector** = mean(all language vectors at Tc), then normalized to unit length.""")
        
        print("DEBUG: About to process UMAP projection")
        
        # UMAP projection if available
        if hasattr(st.session_state, 'simulation_results') and hasattr(st.session_state, 'analysis_results'):
            st.subheader("🗺️ UMAP Projection")

            simulation_results = st.session_state.simulation_results
            analysis_results = st.session_state.analysis_results
            vector_snapshots = simulation_results.get('vector_snapshots', {})
            snapshot_dir = simulation_results.get('snapshot_directory')
            available_snapshot_temps = simulation_results.get('available_snapshot_temperatures', [])
            selected_temp = None

            print(f"DEBUG: vector_snapshots keys: {list(vector_snapshots.keys()) if vector_snapshots else 'None'}")
            print(f"DEBUG: available_snapshot_temps: {available_snapshot_temps}")

            # Add slider if multiple snapshots are available
            if (snapshot_dir and available_snapshot_temps and len(available_snapshot_temps) > 1) or (vector_snapshots and len(vector_snapshots) > 1):
                print("DEBUG: Multiple snapshots available, creating temperature selector")
                
                # Get estimated temperature range from session state if available
                estimated_range = None
                if hasattr(st.session_state, 'estimated_range'):
                    estimated_range = st.session_state.estimated_range
                    print(f"DEBUG: Found estimated_range in session_state: {estimated_range}")
                
                if snapshot_dir and available_snapshot_temps:
                    # Use disk-based snapshots
                    snapshot_temps = available_snapshot_temps
                    print(f"DEBUG: Using disk-based snapshots: {snapshot_temps}")
                else:
                    # Use memory-based snapshots
                    snapshot_temps = sorted(list(vector_snapshots.keys()))
                    print(f"DEBUG: Using memory-based snapshots: {snapshot_temps}")
                
                tc = analysis_results.get('critical_temperature')
                print(f"DEBUG: tc from analysis_results: {tc}")
                
                # Get Tc from session state if not in analysis_results
                if tc is None and hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
                    tc = st.session_state.critical_temperature
                    print(f"DEBUG: tc from session_state: {tc}")
                
                # Use estimated range if available, otherwise use snapshot range
                if estimated_range:
                    est_min, est_max = estimated_range
                    print(f"DEBUG: Using estimated range: min={est_min}, max={est_max}")
                    
                    # Find closest temperatures to estimated range from available snapshots
                    closest_to_est_min = min(snapshot_temps, key=lambda t: abs(t - est_min))
                    closest_to_est_max = min(snapshot_temps, key=lambda t: abs(t - est_max))
                    
                    if tc is not None:
                        # Find closest temperature to Tc from available snapshots
                        closest_to_tc = min(snapshot_temps, key=lambda t: abs(t - tc))
                        display_temps = [closest_to_est_min, closest_to_tc, closest_to_est_max]
                        default_temp = closest_to_tc  # Default to Tc
                        print(f"DEBUG: Using estimated range + Tc: min={closest_to_est_min}, tc={closest_to_tc}, max={closest_to_est_max}")
                    else:
                        display_temps = [closest_to_est_min, closest_to_est_max]
                        default_temp = closest_to_est_min
                        print(f"DEBUG: Using estimated range only: min={closest_to_est_min}, max={closest_to_est_max}")
                else:
                    # Fallback to original logic using snapshot range
                    min_temp = min(snapshot_temps)
                    max_temp = max(snapshot_temps)
                    print(f"DEBUG: Using snapshot range: min={min_temp}, max={max_temp}")
                    
                    if tc is not None:
                        # Find closest temperature to Tc from available snapshots
                        closest_to_tc = min(snapshot_temps, key=lambda t: abs(t - tc))
                        display_temps = [min_temp, closest_to_tc, max_temp]
                        default_temp = closest_to_tc  # Default to Tc
                        print(f"DEBUG: Using snapshot range + Tc: min={min_temp}, tc={closest_to_tc}, max={max_temp}")
                    else:
                        # Fallback if no Tc available
                        display_temps = [min_temp, max_temp]
                        default_temp = min_temp
                        print(f"DEBUG: No Tc available, using min/max only")
                
                # Remove duplicates and sort
                display_temps = sorted(list(set(display_temps)))
                print(f"DEBUG: Final display_temps: {display_temps}")
                
                # Create labels for dropdown
                temp_labels = []
                for temp in display_temps:
                    print(f"DEBUG: Processing temp: {temp}, type: {type(temp)}")
                    if estimated_range and abs(temp - estimated_range[0]) < 0.01:
                        try:
                            label = f"Est Min ({float(temp):.3f})"
                        except (ValueError, TypeError):
                            label = f"Est Min ({temp})"
                    elif estimated_range and abs(temp - estimated_range[1]) < 0.01:
                        try:
                            label = f"Est Max ({float(temp):.3f})"
                        except (ValueError, TypeError):
                            label = f"Est Max ({temp})"
                    elif temp == min(snapshot_temps):
                        try:
                            label = f"Min ({float(temp):.3f})"
                        except (ValueError, TypeError):
                            label = f"Min ({temp})"
                    elif temp == max(snapshot_temps):
                        try:
                            label = f"Max ({float(temp):.3f})"
                        except (ValueError, TypeError):
                            label = f"Max ({temp})"
                    elif tc is not None and abs(temp - tc) < 0.01:
                        try:
                            label = f"Tc ({float(temp):.3f})"
                        except (ValueError, TypeError):
                            label = f"Tc ({temp})"
                    else:
                        try:
                            label = f"{float(temp):.3f}"
                        except (ValueError, TypeError):
                            label = f"{temp}"
                    temp_labels.append(label)
                
                print(f"DEBUG: temp_labels: {temp_labels}")
                
                # Show info about the temperature selection
                if estimated_range:
                    est_min_formatted = f"{float(estimated_range[0]):.3f}"
                    est_max_formatted = f"{float(estimated_range[1]):.3f}"
                    st.info(f"📊 **Temperature Selection**: Using estimated range ({est_min_formatted} - {est_max_formatted}) with available snapshots")
                else:
                    min_temp_formatted = f"{float(min(snapshot_temps)):.3f}"
                    max_temp_formatted = f"{float(max(snapshot_temps)):.3f}"
                    st.info(f"📊 **Temperature Selection**: Using snapshot range ({min_temp_formatted} - {max_temp_formatted})")

                # Use dropdown instead of slider
                # Determine default index (prefer Tc if available)
                default_temp = tc if tc is not None else display_temps[0]
                
                # Find the label for the default temperature
                default_label = None
                if tc is not None and abs(default_temp - tc) < 0.01:
                    try:
                        default_label = f"Tc ({float(default_temp):.3f})"
                    except (ValueError, TypeError):
                        default_label = f"Tc ({default_temp})"
                else:
                    try:
                        default_label = f"{float(default_temp):.3f}"
                    except (ValueError, TypeError):
                        default_label = f"{default_temp}"
                
                # Find the index of the default label
                try:
                    default_index = temp_labels.index(default_label)
                except ValueError:
                    default_index = 0  # Fallback to first option
                
                print(f"DEBUG: About to create selectbox with default_index: {default_index}")
                selected_label = st.selectbox(
                    "Select Temperature for UMAP",
                    options=temp_labels,
                    index=default_index,
                    help="Select temperature to see how semantic structure evolves. The plot dynamically updates."
                )
                
                selected_temp = display_temps[temp_labels.index(selected_label)]
                print(f"DEBUG: selected_temp: {selected_temp}")

            # Load the vector data for the selected temperature
            snapshot_data = None
            if snapshot_dir:
                snapshot_data = _load_snapshot_from_disk(snapshot_dir, selected_temp)
            
            vectors_to_plot = snapshot_data.get('vectors') if snapshot_data else None

            if vectors_to_plot is not None:
                try:
                    anchor_language = experiment_config.get('anchor_language')
                    include_anchor = experiment_config.get('include_anchor', False)
                    
                    # Get dynamics languages from multiple sources
                    dynamics_languages = simulation_results.get('dynamics_languages', [])
                    
                    # If dynamics_languages is empty, try to get from snapshot data
                    if not dynamics_languages and snapshot_data:
                        dynamics_languages = snapshot_data.get('languages', [])
                    
                    # If still empty, use the languages from experiment_config
                    if not dynamics_languages:
                        dynamics_languages = experiment_config.get('dynamics_languages', [])
                    
                    # If still empty, create default language codes based on vector count
                    if not dynamics_languages:
                        n_vectors = len(vectors_to_plot)
                        dynamics_languages = [f"lang_{i}" for i in range(n_vectors)]
                        print(f"DEBUG: Created default language codes for {n_vectors} vectors")
                    
                    print(f"DEBUG: Using dynamics_languages: {dynamics_languages[:5]}... (total: {len(dynamics_languages)})")
                    print(f"DEBUG: Calling plot_full_umap_projection with vectors of shape {vectors_to_plot.shape}")
                    
                    fig = plot_full_umap_projection(
                        vectors_at_temp=vectors_to_plot,
                        dynamics_languages=dynamics_languages,
                        analysis_results=analysis_results,
                        anchor_language=anchor_language,
                        include_anchor=include_anchor,
                        target_temp=selected_temp
                    )
                    st.plotly_chart(fig, use_container_width=True, key="umap_projection_chart")
                except Exception as e:
                    print(f"DEBUG: Exception in UMAP projection plotting: {e}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    st.error("An error occurred while rendering the UMAP projection.")
            else:
                st.warning(f"No vector data available for the selected temperature (T={selected_temp:.3f}). Cannot display UMAP plot.")
                
            # Display interactive metrics plot below UMAP, using the same loaded data
            if selected_temp is not None:
                st.subheader("Interactive Metrics at Selected Temperature")
                display_interactive_metrics_at_temperature(
                    simulation_results, 
                    analysis_results, 
                    selected_temp, 
                    experiment_config.get('anchor_language'), 
                    experiment_config.get('include_anchor', False)
                )
        
        print("DEBUG: render_anchor_comparison_tab completed successfully")
        
    except Exception as e:
        print(f"DEBUG: Exception in render_anchor_comparison_tab: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        logger.error(f"Error displaying comparison metrics: {e}")
        st.error("Error displaying comparison metrics. Please try again.")


def display_interactive_metrics_at_temperature(simulation_results: Dict[str, Any], 
                                             analysis_results: Dict[str, Any], 
                                             selected_temperature: float,
                                             anchor_language: str,
                                             include_anchor: bool) -> None:
    """
    Display interactive metrics for the selected temperature
    
    Args:
        simulation_results: Simulation results dictionary
        analysis_results: Analysis results dictionary
        selected_temperature: Selected temperature for analysis
        anchor_language: Anchor language code
        include_anchor: Whether anchor is included in dynamics
    """
    try:
        snapshot_dir = simulation_results.get('snapshot_directory')
        
        vectors_at_temp = None
        if snapshot_dir:
            snapshot_data = _load_snapshot_from_disk(snapshot_dir, selected_temperature)
            if snapshot_data:
                vectors_at_temp = snapshot_data.get('vectors')

        if vectors_at_temp is None:
            st.warning(f"No vectors available at temperature {selected_temperature:.3f}")
            return
        
        # --- Metrics Calculation ---
        # 1. Anchor vs Meta-vector Cosine Similarity
        from core.comparison_metrics import compute_cosine_similarity_matrix, compute_meta_vector
        
        # Calculate meta-vector (centroid of all vectors at this temperature)
        meta_result = compute_meta_vector(vectors_at_temp)
        meta_vector = meta_result['meta_vector']
        
        # Get anchor vector - try multiple sources
        anchor_vector = None
        
        # First, try to get anchor vector from session state
        if hasattr(st.session_state, 'anchor_vectors') and st.session_state.anchor_vectors is not None:
            anchor_vector = st.session_state.anchor_vectors[0] if len(st.session_state.anchor_vectors) > 0 else None
            print(f"DEBUG: Got anchor vector from session_state.anchor_vectors")
        
        # If not in session state, try to get from analysis results
        if anchor_vector is None and analysis_results and 'anchor_vectors' in analysis_results:
            anchor_vector = analysis_results['anchor_vectors'][0] if len(analysis_results['anchor_vectors']) > 0 else None
            print(f"DEBUG: Got anchor vector from analysis_results.anchor_vectors")
        
        # If still not found, try to get from simulation results
        if anchor_vector is None and 'anchor_vector' in simulation_results:
            anchor_vector = simulation_results['anchor_vector'][0] if len(simulation_results['anchor_vector']) > 0 else None
            print(f"DEBUG: Got anchor vector from simulation_results.anchor_vector")
        
        # If still not found and anchor is included in dynamics, we need to get the original anchor vector
        # The first vector in vectors_at_temp is the evolved anchor vector, not the original
        if anchor_vector is None and include_anchor:
            # Try to get original anchor vector from session state
            if hasattr(st.session_state, 'original_anchor_vector') and st.session_state.original_anchor_vector is not None:
                anchor_vector = st.session_state.original_anchor_vector
                print(f"DEBUG: Got original anchor vector from session_state")
            elif analysis_results and 'original_anchor_vector' in analysis_results:
                anchor_vector = analysis_results['original_anchor_vector']
                print(f"DEBUG: Got original anchor vector from analysis_results")
            else:
                # Fallback: use the first vector but warn that it's evolved
                anchor_vector = vectors_at_temp[0]
                print(f"DEBUG: WARNING - Using evolved anchor vector (first vector) as fallback")
        
        # If anchor is not included in dynamics, we need to load it separately
        if anchor_vector is None and not include_anchor:
            # Try to get anchor vector from session state or analysis results
            if hasattr(st.session_state, 'original_anchor_vector') and st.session_state.original_anchor_vector is not None:
                anchor_vector = st.session_state.original_anchor_vector
                print(f"DEBUG: Got original anchor vector from session_state")
            elif analysis_results and 'original_anchor_vector' in analysis_results:
                anchor_vector = analysis_results['original_anchor_vector']
                print(f"DEBUG: Got original anchor vector from analysis_results")
            else:
                st.warning("Anchor vector not available for comparison (anchor excluded from dynamics)")
                return
        
        if anchor_vector is None:
            st.warning("Could not determine anchor vector for comparison")
            return
        
        print(f"DEBUG: Anchor vector shape: {anchor_vector.shape if hasattr(anchor_vector, 'shape') else 'scalar'}")
        print(f"DEBUG: Meta vector shape: {meta_vector.shape}")
        
        # Calculate cosine similarity between anchor and meta-vector
        from sklearn.metrics.pairwise import cosine_similarity
        anchor_meta_similarity = cosine_similarity([anchor_vector], [meta_vector])[0][0]
        
        # Calculate cosine distance (1 - similarity)
        anchor_meta_distance = 1 - anchor_meta_similarity
        
        print(f"DEBUG: Anchor-Meta Similarity: {anchor_meta_similarity:.4f}")
        print(f"DEBUG: Anchor-Meta Distance: {anchor_meta_distance:.4f}")

        # 2. Clustering - Use Ising-compatible clustering for consistency with UMAP plot
        from core.clustering import ising_compatible_clustering
        
        # Get critical temperature from analysis results or session state
        critical_temperature = None
        if analysis_results and 'critical_temperature' in analysis_results:
            critical_temperature = analysis_results['critical_temperature']
        elif hasattr(st, 'session_state') and hasattr(st.session_state, 'critical_temperature'):
            critical_temperature = st.session_state.critical_temperature
        
        # Use the same clustering approach as UMAP plot
        clustering_result = ising_compatible_clustering(
            vectors_at_temp, 
            temperature=selected_temperature,
            critical_temperature=critical_temperature,
            min_cluster_size=2
        )
        
        num_clusters = clustering_result['n_clusters']
        cluster_labels = clustering_result['cluster_labels']
        
        print(f"DEBUG: Number of clusters: {num_clusters}")
        print(f"DEBUG: Cluster labels: {cluster_labels}")
        
        # Calculate silhouette score using the cluster labels from Ising clustering
        from sklearn.metrics import silhouette_score
        try:
            if num_clusters > 1 and len(np.unique(cluster_labels)) > 1:
                silhouette_score_val = silhouette_score(vectors_at_temp, cluster_labels)
            else:
                silhouette_score_val = 0.0
        except ValueError:
            silhouette_score_val = 0.0

        # 3. Alignment - get from simulation results
        temperatures = simulation_results.get('temperatures', [])
        alignment_at_temp = np.nan
        if len(temperatures) > 0:
            temp_idx = np.argmin(np.abs(np.array(temperatures) - selected_temperature))
            alignment = simulation_results.get('alignment', [])
            if len(alignment) > temp_idx:
                alignment_at_temp = alignment[temp_idx]

        # --- Display Metrics in 4 columns ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Anchor-Meta Similarity", 
                f"{anchor_meta_similarity:.4f}",
                help="Cosine similarity between anchor language vector and meta-vector (centroid of all language vectors) at the selected temperature. Range: 0-1. Higher values indicate the anchor language aligns better with the emergent multilingual semantic structure."
            )
        
        with col2:
            st.metric(
                "Anchor-Meta Distance", 
                f"{anchor_meta_distance:.4f}",
                help="Cosine distance between anchor language vector and meta-vector (1 - similarity). Range: 0-1. Lower values indicate the anchor language is closer to the emergent multilingual semantic structure."
            )

        with col3:
            st.metric(
                "Number of Clusters", 
                f"{num_clusters}",
                help="Number of distinct semantic groups identified by Ising-compatible clustering at this temperature. Uses adaptive, temperature-dependent thresholds. Fewer clusters suggest more semantic convergence across languages."
            )

        with col4:
            st.metric(
                "Silhouette Score", 
                f"{silhouette_score_val:.4f}",
                help="Measures cluster quality using Ising clustering results. Range: -1 to 1. Higher values indicate well-separated, cohesive clusters. 0.0 when only one cluster exists."
            )
        
        # Add alignment metric in a separate row
        if not np.isnan(alignment_at_temp):
            st.metric(
                "System Alignment", 
                f"{alignment_at_temp:.4f}",
                help="Average cosine similarity across all language vectors at this temperature (legacy metric). Range: 0-1. Higher values indicate overall semantic convergence in the system."
            )

    except Exception as e:
        logger.error(f"Error displaying interactive metrics at T={selected_temperature}: {e}", exc_info=True)
        st.warning(f"Could not display interactive metrics: {e}")


def display_comparison_metrics(comparison_metrics: Dict[str, float]) -> None:
    """
    Display comparison metrics in organized format
    
    Args:
        comparison_metrics: Dictionary of comparison metrics
    """
    try:
        st.subheader("Comparison Metrics")
        
        # Helper function to safely format metric values
        def safe_format_metric(value, default=0.0):
            try:
                if value is None:
                    return f"{default:.4f}"
                # Convert to float and format
                return f"{float(value):.4f}"
            except (ValueError, TypeError):
                return f"{default:.4f}"
        
        # Primary metric: Cosine Distance
        cosine_distance = comparison_metrics.get('cosine_distance', 0.0)
        st.metric(
            "Cosine Distance",
            safe_format_metric(cosine_distance),
            help="Distance between anchor language vector and multilingual meta-vector at Tc. Lower values indicate more similar meaning."
        )
        
        # Supporting metric: Cosine Similarity
        cosine_similarity = comparison_metrics.get('cosine_similarity', 0.0)
        st.metric(
            "Cosine Similarity",
            safe_format_metric(cosine_similarity),
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
            # Check if we have comparison data available
            if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
                comparison_metrics = st.session_state.analysis_results.get('anchor_comparison', {})
                print(f"DEBUG: Session state analysis_results keys: {list(st.session_state.analysis_results.keys())}")
                print(f"DEBUG: Session state anchor_comparison: {st.session_state.analysis_results.get('anchor_comparison', {})}")
                print(f"DEBUG: Session state critical_temperature: {st.session_state.analysis_results.get('critical_temperature', 'Not found')}")
                
                # Use actual languages from session state if available, otherwise fallback to hardcoded list
                anchor_language = comparison_metrics.get('anchor_language', 'Unknown')
                include_anchor = comparison_metrics.get('include_anchor', False)
                
                # Display comparison metrics
                display_comparison_metrics(comparison_metrics)
                
                # Display comparison visualization
                display_comparison_visualization(comparison_metrics)
            else:
                st.warning("No comparison data available in session state")
        
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
                
                # Safely convert value to float for formatting
                try:
                    float_value = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    float_value = 0.0
                
                st.write(f"### {metric_name.replace('_', ' ').title()}")
                try:
                    value_formatted = f"{float_value:.4f}"
                except (ValueError, TypeError):
                    value_formatted = f"{float_value}"
                st.write(f"**Value:** {value_formatted}")
                st.write(f"**Description:** {info['description']}")
                
                # Determine quality level
                if metric_name in ['cosine_similarity']:
                    # Higher is better for these metrics
                    if float_value > 0.8:
                        quality = "excellent"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif float_value > 0.6:
                        quality = "good"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif float_value > 0.4:
                        quality = "moderate"
                        st.warning(f"**Quality:** {quality} ({info[quality]})")
                    else:
                        quality = "poor"
                        st.error(f"**Quality:** {quality} ({info[quality]})")
                else:
                    # Lower is better for these metrics
                    if float_value < 0.1:
                        quality = "excellent"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif float_value < 0.3:
                        quality = "good"
                        st.success(f"**Quality:** {quality} ({info[quality]})")
                    elif float_value < 0.6:
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
        
        # Safely convert to float
        try:
            directional_score = float(cosine_similarity) if cosine_similarity is not None else 0.0
        except (ValueError, TypeError):
            directional_score = 0.0
        
        st.write("### Composite Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                directional_score_formatted = f"{directional_score:.3f}"
            except (ValueError, TypeError):
                directional_score_formatted = f"{directional_score}"
            st.metric("Directional Similarity", directional_score_formatted)
        
        with col2:
            try:
                overall_score_formatted = f"{directional_score:.3f}"
            except (ValueError, TypeError):
                overall_score_formatted = f"{directional_score}"
            st.metric("Overall Similarity", overall_score_formatted)
        
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
                try:
                    float_value = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    float_value = 0.0
                
                if float_value > 0.7:
                    high_similarity_metrics.append(metric_name)
                elif float_value < 0.4:
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
        
        # Safely convert values to float
        safe_values = []
        for value in values:
            try:
                safe_values.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                safe_values.append(0.0)
        
        # Normalize values to 0-1 scale for radar chart
        normalized_values = []
        for i, (name, value) in enumerate(zip(metric_names, safe_values)):
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
        
        # Safely format values
        formatted_values = []
        for value in safe_values:
            try:
                formatted_values.append(f"{value:.4f}")
            except (ValueError, TypeError):
                formatted_values.append(f"{value}")
        
        formatted_normalized = []
        for norm in normalized_values:
            try:
                formatted_normalized.append(f"{norm:.3f}")
            except (ValueError, TypeError):
                formatted_normalized.append(f"{norm}")
        
        data = {
            'Metric': [name.replace('_', ' ').title() for name in metric_names],
            'Value': formatted_values,
            'Normalized': formatted_normalized
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
        print("DEBUG: Starting display_umap_projection function")
        
        if (hasattr(st.session_state, 'simulation_results') and 
            hasattr(st.session_state, 'analysis_results')):
            
            print("DEBUG: Session state has simulation_results and analysis_results")
            simulation_results = st.session_state.simulation_results
            analysis_results = st.session_state.analysis_results
            
            print(f"DEBUG: simulation_results keys: {list(simulation_results.keys()) if simulation_results else 'None'}")
            print(f"DEBUG: analysis_results keys: {list(analysis_results.keys()) if analysis_results else 'None'}")
            
            if (simulation_results and analysis_results and 
                'vector_snapshots' in simulation_results):
                
                print("DEBUG: Found vector_snapshots in simulation_results")
                st.subheader("🗺️ UMAP Projection")
                
                # Extract anchor language information from experiment config
                anchor_language = experiment_config.get('anchor_language') if experiment_config else None
                include_anchor = experiment_config.get('include_anchor', False) if experiment_config else False
                
                print(f"DEBUG: anchor_language: {anchor_language}, include_anchor: {include_anchor}")
                
                print("DEBUG: About to call plot_full_umap_projection")
                fig = plot_full_umap_projection(
                    simulation_results, 
                    analysis_results,
                    anchor_language=anchor_language,
                    include_anchor=include_anchor
                )
                print("DEBUG: plot_full_umap_projection completed successfully")
                
                if fig.data:  # Check if figure has data
                    print("DEBUG: Figure has data, displaying plotly chart")
                    st.plotly_chart(fig, use_container_width=True, key="umap_projection_display_chart")
                else:
                    print("DEBUG: Figure has no data")
                    st.info("UMAP projection not available (vector snapshots may not be stored)")
            else:
                print("DEBUG: Missing vector_snapshots in simulation_results")
                st.info("UMAP projection requires vector snapshots from simulation")
        else:
            print("DEBUG: Missing simulation_results or analysis_results in session state")
            st.info("Run simulation first to see UMAP projection")
            
    except Exception as e:
        print(f"DEBUG: Exception in display_umap_projection: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
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
        
        # Safely convert to float
        try:
            cosine_similarity_float = float(cosine_similarity) if cosine_similarity is not None else 0.0
        except (ValueError, TypeError):
            cosine_similarity_float = 0.0
        
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
        if cosine_similarity_float > 0.7:
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
        elif cosine_similarity_float > 0.5:
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
        
        # Safely convert metric values to float
        safe_metric_values = []
        for value in metric_values:
            try:
                safe_metric_values.append(float(value) if value is not None else 0.0)
            except (ValueError, TypeError):
                safe_metric_values.append(0.0)
        
        metric_range = max(safe_metric_values) - min(safe_metric_values)
        
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