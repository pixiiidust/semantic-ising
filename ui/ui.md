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
- `render_power_law_summary(power_law_data)` - Display power law analysis
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
**Purpose**: Interactive chart generation using Plotly

**Key Functions**:
- `plot_entropy_vs_temperature(simulation_results)` - Entropy vs temperature plot
- `plot_full_umap_projection(simulation_results, analysis_results, anchor_language=None)` - UMAP projection
- `plot_correlation_decay(analysis_results)` - Correlation decay plot
- `plot_correlation_length_vs_temperature(simulation_results)` - Correlation length vs temperature
- `plot_alignment_vs_temperature(simulation_results)` - Alignment vs temperature
- `plot_energy_vs_temperature(simulation_results)` - Energy vs temperature

**Features**:
- Interactive Plotly charts with hover information
- Critical temperature markers
- Anchor language highlighting in UMAP
- Configurable styling and themes
- Export functionality

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
- Temperature range configuration
- Anchor language configuration
- Real-time simulation monitoring
- Results display with metrics
- Export functionality integration

### 🔗 `tabs/anchor_comparison.py`
**Purpose**: Anchor language comparison analysis

**Key Functions**:
- `render_anchor_comparison_tab()` - Anchor comparison interface

**Features**:
- UMAP visualization with anchor highlighting
- Comprehensive comparison metrics display
- Automatic interpretation and scoring
- Interactive chart exploration
- Export functionality

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

### Custom Chart Configuration
```python
from ui.charts import plot_full_umap_projection

# Create UMAP plot with anchor highlighting
fig = plot_full_umap_projection(
    simulation_results, 
    analysis_results, 
    anchor_language="en"
)

# Customize display
fig.update_layout(
    title="UMAP Projection at Critical Temperature",
    width=800,
    height=600
)
```

## 🎨 Styling and Themes

### Consistent Design
- **Color Scheme**: Consistent color palette across all components
- **Typography**: Unified font and text styling
- **Layout**: Responsive design with proper spacing
- **Icons**: Emoji-based visual indicators

### Interactive Elements
- **Sidebar Configuration**: Clean parameter controls
- **Progress Indicators**: Real-time simulation progress
- **Status Messages**: Clear feedback for user actions
- **Export Buttons**: Easy access to result downloads

## 🧪 Testing

UI components have comprehensive test coverage:

- **Component Tests**: Individual component functionality
- **Integration Tests**: Tab and chart integration
- **Visual Tests**: Chart generation and styling
- **Error Handling**: Graceful error display

Run UI tests with:
```bash
pytest tests/test_ui.py -v
pytest tests/test_ui_integration.py -v
```

## 📱 User Experience

### Workflow Design
1. **Overview Tab**: Learn about the simulator and scientific background
2. **Simulation Tab**: Configure and run experiments
3. **Anchor Comparison Tab**: Analyze results and anchor relationships

### Accessibility Features
- **Clear Navigation**: Intuitive tab structure
- **Helpful Tooltips**: Context-sensitive information
- **Error Messages**: Clear feedback for issues
- **Export Options**: Multiple format support

### Performance Optimization
- **Lazy Loading**: Charts generated on demand
- **Caching**: Simulation results cached for reuse
- **Memory Management**: Efficient data handling
- **Responsive Design**: Works on different screen sizes

## 📚 References

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **UMAP**: Dimensionality reduction algorithm
- **Material Design**: UI/UX design principles 

## 📝 Recent UI/UX Updates
- Section headers and chart explanations updated for clarity and professionalism.
- Convergence summary chart now only shows the vertical Tc line (critical temperature), with the convergence threshold line removed.
- Anchor comparison tab is cleaner, with collapsible sections and no redundant metrics.
- All debug output removed from user-facing UI.
- UI and backend now always use the same Tc value for both display and charting, ensuring perfect synchronization. 