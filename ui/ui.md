# UI Module Documentation

The `ui/` directory contains the Streamlit-based user interface components for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Tabs](#tabs)
- [Charts](#charts)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The UI modules provide an interactive web interface for the Semantic Ising Simulator, featuring:

- **Streamlit-based interface** with real-time simulation monitoring
- **Interactive visualizations** using Plotly charts
- **Anchor language configuration** with single-phase vs two-phase modes
- **Comprehensive result display** with metrics and interpretations
- **Export functionality** for results and visualizations

## 📦 Components

### 🎛️ `components.py`
**Purpose**: Reusable UI components and configuration widgets

**Key Functions**:
- `render_anchor_config()` - Anchor language configuration sidebar
- `render_experiment_description(anchor_language, include_anchor, dynamics_languages)` - Display experiment details
- `render_metrics_summary(analysis_results)` - Display key metrics
- `render_critical_temperature_display(tc)` - Display critical temperature
- `render_anchor_comparison_summary(comparison_metrics)` - Display anchor comparison
- `render_export_buttons(simulation_results, analysis_results)` - Export functionality
- `render_error_message(message)` - Error display
- `render_success_message(message)` - Success display
- `render_warning_message(message)` - Warning display

**Features**:
- Consistent styling and layout
- Real-time configuration updates
- Comprehensive error handling
- Export functionality integration

### 📊 `charts.py`
**Purpose**: Data visualization and plotting utilities

**Key Functions**:
- `plot_correlation_length(temperatures, correlation_lengths, tc)` - Correlation length visualization
- `plot_umap_embedding(vectors, labels, title)` - UMAP dimensionality reduction plots
- `plot_metrics_over_temperature(temperatures, metrics, tc)` - Temperature-dependent metrics

**Features**:
- **Critical temperature annotation**: Tc marked on all relevant plots
- **Interactive UMAP**: Dynamic visualization with temperature slider
- **Multi-panel layouts**: Comprehensive analysis views
- **Language code preservation**: Actual language codes displayed (en, es, fr, etc.)
- **Responsive design**: Adapts to different screen sizes

## 📑 Tabs

### 🏠 `tabs/overview.py`
**Purpose**: Introduction and instructions tab

**Key Functions**:
- `render_overview_tab()` - Display overview content

**Features**:
- Project introduction and scientific background
- Step-by-step usage instructions
- Interactive accordion sections
- Links to documentation and examples

### ⚙️ `tabs/simulation.py`
**Purpose**: Main simulation interface with configuration and results

**Key Functions**:
- `render_simulation_tab()` - Main simulation interface

**Features**:
- Concept and encoder selection
- Temperature range configuration with auto-estimation
- Anchor language configuration
- Real-time simulation monitoring with progress bars
- Results display with 3-tab structure:
  - **📈 Metrics**: Main simulation metrics (alignment, entropy, energy, correlation length)
  - **🔄 Convergence**: Convergence analysis and entropy vs correlation length
  - **📋 Details**: Simulation configuration and parameters
- Export functionality integration

### 🔗 `tabs/anchor_comparison.py`
**Purpose**: Anchor language comparison analysis with interactive temperature exploration

**Key Functions**:
- `render_anchor_comparison_tab()` - Anchor comparison interface

**Features**:
- **Interactive UMAP visualization** with temperature slider for dynamic exploration
- **Temperature-based vector loading** from disk snapshots for efficient memory usage
- **Comprehensive comparison metrics display** in three-column layout:
  - Critical Temperature (Tc) prominently displayed
  - Cosine Distance (primary semantic metric)
  - Cosine Similarity (directional similarity)
- **Automatic interpretation and scoring** of anchor alignment
- **Pre-calculated metrics** for all temperatures to avoid real-time computation
- **Language code preservation** in UMAP plots (en, es, fr, etc.)
- **Export functionality** for results and visualizations
- **Consistent metrics calculation**: Both main metrics and interactive metrics use original anchor vectors for comparison
- **Disk-based recalculation**: Proper handling of disk-based snapshots when in-memory snapshots are not available
- **Session state synchronization**: All UI components display consistent metrics by updating session state after recalculation

### 📋 `tabs/tabs.md`
**Purpose**: Tab structure documentation

**Content**:
- Tab organization and navigation
- Component dependencies
- User workflow guidance

## 🎨 Chart Features

### Interactive Visualizations
- **Hover Information**: Detailed data points on mouse hover
- **Zoom and Pan**: Interactive chart navigation
- **Legend Controls**: Toggle visibility of data series
- **Export Options**: Save charts as PNG/SVG

### Critical Temperature Markers
- **Vertical Lines**: Mark Tc on temperature-dependent plots
- **Annotations**: Clear labeling of critical points
- **Color Coding**: Consistent visual identification

### Anchor Language Highlighting
- **Special Markers**: Red stars for anchor languages in UMAP
- **Color Coding**: Distinct colors for anchor vs dynamics languages
- **Hover Information**: Language codes and distances

## 🔗 Dependencies

### Internal Dependencies
```
ui/
├── components.py (independent)
├── charts.py (independent)
└── tabs/
    ├── overview.py (depends on components.py)
    ├── simulation.py (depends on components.py, charts.py)
    ├── anchor_comparison.py (depends on charts.py)
    └── tabs.md (documentation)
```

### External Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive charting library
- **NumPy**: Numerical computations
- **UMAP**: Dimensionality reduction
- **Pandas**: Data manipulation

## 💡 Usage Examples

### Basic UI Setup
```python
import streamlit as st
from ui.components import render_anchor_config
from ui.charts import plot_entropy_vs_temperature

# Configure anchor language
anchor_lang, include_anchor = render_anchor_config()

# Display chart
if simulation_results:
    fig = plot_entropy_vs_temperature(simulation_results)
    st.plotly_chart(fig, use_container_width=True)
```

### Tab Integration
```python
from ui.tabs.simulation import render_simulation_tab
from ui.tabs.anchor_comparison import render_anchor_comparison_tab

# Main app structure
tab1, tab2, tab3 = st.tabs(["Overview", "Simulation", "Anchor Comparison"])

with tab1:
    render_overview_tab()

with tab2:
    render_simulation_tab()

with tab3:
    render_anchor_comparison_tab()
```

## 📝 Recent Updates

### Interactive UMAP Visualization
- **Temperature slider integration**: Added interactive temperature slider for dynamic UMAP exploration
- **Disk-based snapshot loading**: Efficient loading of vector snapshots from disk based on temperature selection
- **UMAP zoom control**: Auto-scaling with 2.0× zoom factor for better visualization of language clusters
- **Language code preservation**: Fixed UMAP labels to show actual language codes (en, es, fr, etc.) instead of generic labels

### Enhanced Metrics Display
- **Three-column layout**: Critical Temperature, Cosine Distance, and Cosine Similarity displayed consistently
- **Pre-calculated metrics**: All temperature metrics calculated after simulation to avoid real-time computation
- **Improved Tc display**: Critical temperature prominently shown with enhanced help text
- **Debug output cleanup**: Removed debug messages for cleaner user experience

### Performance Optimizations
- **Memory efficiency**: Temperature-based snapshot loading reduces memory usage for large simulations
- **Real-time responsiveness**: Pre-calculated metrics enable smooth temperature slider interaction
- **Efficient data retrieval**: Hash-based snapshot directory naming for unique simulation configurations