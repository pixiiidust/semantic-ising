"""
Overview tab for the Semantic Ising Simulator
Provides information about the simulator and usage instructions
"""

import streamlit as st
from typing import List


def render_overview_tab(concept: str, encoder: str, T_range: List[float], 
                       anchor_language: str, include_anchor: bool):
    """
    Render the overview tab with information about the simulator
    """
    st.header("📋 Semantic Ising Simulator Overview")
    
    # What does this simulator do?
    with st.expander("🔬 What does this simulator do?", expanded=True):
        st.markdown("""
        This simulator tests whether semantically identical words across languages converge in embedding space under Ising dynamics.
        
        **Core Question**: Do words meaning "dog" in 70+ languages share a common latent semantic structure?
        
        **Method**: 
        - Load multilingual embeddings for a concept (e.g., "dog")
        - Apply Ising-style Monte Carlo dynamics at different temperatures
        - Detect critical temperature (Tc) where semantic phase transition occurs
        - Analyze convergence patterns and anchor language comparison
        """)
    
    # What's new in this version?
    with st.expander("🆕 What's new in this version?", expanded=False):
        st.markdown("""
        **Version 2.0 Features**:
        - ✅ **Anchor Language Configuration**: Choose which language participates in dynamics vs comparison
        - ✅ **Vector Interpolation**: Get vectors at exactly Tc, not just nearby snapshots
        - ✅ **Advanced Comparison Metrics**: Procrustes distance, CKA similarity, EMD distance
        - ✅ **Two-Phase Experiments**: Compare anchor language to emergent multilingual structure
        - ✅ **Enhanced UI**: Better charts, clearer explanations, improved error handling
        """)
    
    # Step-by-step instructions
    with st.expander("📝 How to use this simulator", expanded=False):
        st.markdown("""
        **Step-by-Step Process**:
        
        1️⃣ **Configure Experiment** (Sidebar)
        - Select concept (e.g., "dog", "tree", "house")
        - Choose encoder model (LaBSE recommended)
        - Set temperature range and steps
        - Configure anchor language settings
        
        2️⃣ **Run Simulation** (Simulation Results tab)
        - Click "Run Simulation" button
        - Watch progress bar and real-time metrics
        - System will detect critical temperature (Tc)
        
        3️⃣ **Analyze Results** (Simulation Results tab)
        - Examine alignment, entropy, and correlation curves
        - Look for phase transition at Tc
        - Check UMAP projections for semantic clustering
        
        4️⃣ **Compare Anchor** (Anchor Language Comparison tab)
        - View detailed comparison metrics
        - Understand anchor language relationship to multilingual structure
        - Export results for further analysis
        """)
    
    # Current configuration
    with st.expander("⚙️ Current Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Concept", concept)
            st.metric("Encoder", encoder)
            st.metric("Temperature Range", f"{T_range[0]:.1f} - {T_range[-1]:.1f}")
        with col2:
            st.metric("Temperature Steps", len(T_range))
            st.metric("Anchor Language", anchor_language)
            st.metric("Experiment Type", "Single-phase" if include_anchor else "Two-phase")
    
    # Scientific background
    with st.expander("🧠 Scientific Background", expanded=False):
        st.markdown("""
        **Theoretical Foundation**:
        
        **Ising Model**: Statistical physics model for phase transitions
        - Vectors represent "spins" in semantic space
        - Temperature controls randomness vs alignment
        - Critical temperature marks phase transition
        
        **Semantic Phase Transition**:
        - Low T: Vectors align (ordered phase)
        - High T: Vectors randomize (disordered phase)
        - Tc: Critical point where transition occurs
        
        **Multilingual Hypothesis**:
        - Semantically identical words should converge at Tc
        - Reveals shared latent semantic structure
        - Tests universal semantic representation hypothesis
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues**:
        
        **PyTorch Import Errors**: 
        - Use Python 3.11 or 3.10 for real experiments
        - Current setup uses mock embeddings for testing
        
        **Simulation Divergence**:
        - Normal at high temperatures
        - System automatically handles and reports
        
        **Empty Charts**:
        - Check if simulation has completed
        - Ensure sufficient data for visualization
        
        **Performance**:
        - Large temperature ranges may take time
        - Progress bar shows completion status
        """)
    
    # Quick start
    st.subheader("🚀 Quick Start")
    st.info("""
    **Ready to begin?** 
    
    1. Check your configuration in the sidebar
    2. Go to the "⚙️ Simulation Results" tab
    3. Click "Run Simulation" to start
    4. Watch the magic happen! 🪄
    """) 