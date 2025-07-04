# UI Tabs Documentation

The `ui/tabs/` directory contains tabbed interface components for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Tab Structure](#tab-structure)
- [Tab Components](#tab-components)
- [Navigation](#navigation)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The tabs directory contains the main interface components for the Streamlit application, providing:

- **Overview tab** with project introduction and instructions
- **Simulation tab** with configuration and results display
- **Anchor comparison tab** with UMAP visualization and analysis
- **Modular design** for easy maintenance and extension

## 📁 Tab Structure

```
ui/tabs/
├── overview.py                    # Introduction and instructions tab
├── simulation.py                  # Main simulation interface
├── anchor_comparison.py           # Anchor language comparison
└── tabs.md                        # This documentation file
```

## 🏠 Tab Components

### 📖 `overview.py`
**Purpose**: Introduction and instructions tab

**Key Functions**:
- `render_overview_tab()` - Display overview content

**Features**:
- Project introduction and scientific background
- Step-by-step usage instructions
- Interactive accordion sections
- Links to documentation and examples
- Scientific context and methodology

**Content Sections**:
- **What is this simulator?** - Project overview
- **Scientific Background** - Ising model explanation
- **How to use** - Step-by-step instructions
- **Interpretation Guide** - Understanding results

### ⚙️ `simulation.py`
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

**Configuration Options**:
- **Concept Selection**: Choose from available concepts (dog, tree)
- **Encoder Selection**: Choose embedding model (LaBSE, SBERT)
- **Temperature Range**: Configure simulation temperature sweep with auto-estimation
- **Anchor Configuration**: Set anchor language and inclusion mode

**Results Display**:
- **Metrics Charts**: Alignment, entropy, energy, correlation length
- **Critical Temperature**: Detection and display
- **Convergence Analysis**: Convergence status and entropy vs correlation length
- **Export Options**: CSV, vectors, charts, configuration

### 🔗 `anchor_comparison.py`
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

**Visualization Components**:
- **UMAP Projection**: 2D visualization of vectors with temperature slider control
- **Anchor Highlighting**: Red star markers for anchor languages
- **Hover Information**: Language codes and distances
- **Interactive Controls**: Zoom, pan, temperature selection, and language filtering
- **Temperature Slider**: Dynamic exploration of semantic structure across temperature steps

**Comparison Metrics**:
- **Critical Temperature (Tc)**: Prominently displayed with enhanced help text
- **Cosine Distance**: Primary semantic distance metric (0-1, lower is better)
- **Cosine Similarity**: Directional similarity (0-1, higher is better)

## 🧭 Navigation

### Tab Organization
1. **Overview Tab**: Learn about the simulator and scientific background
2. **Simulation Tab**: Configure and run experiments
3. **Anchor Comparison Tab**: Analyze results and anchor relationships

### User Workflow
1. **Start with Overview**: Understand the project and methodology
2. **Configure Simulation**: Set up experiment parameters
3. **Run Simulation**: Execute the semantic Ising model
4. **Analyze Results**: Explore visualizations and metrics
5. **Export Data**: Download results for further analysis

### Tab Dependencies
```
overview.py (independent)
    ↓
simulation.py (depends on core simulation)
    ↓
anchor_comparison.py (depends on simulation results)
```

## 💡 Usage Examples

### Basic Tab Integration
```python
import streamlit as st
from ui.tabs.overview import render_overview_tab
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

### Custom Tab Configuration
```python
# Custom tab with specific parameters
def render_custom_simulation_tab(concept="dog", encoder="LaBSE"):
    """Render simulation tab with custom defaults."""
    st.header("Custom Simulation")
    
    # Override default parameters
    st.session_state.default_concept = concept
    st.session_state.default_encoder = encoder
    
    # Render standard simulation tab
    render_simulation_tab()
```

### Tab State Management
```python
# Manage state across tabs
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Pass state between tabs
if st.session_state.simulation_results:
    render_anchor_comparison_tab()
else:
    st.warning("Run simulation first to see anchor comparison")
```

## 🎨 Styling and Themes

### Consistent Design
- **Color Scheme**: Consistent color palette across all tabs
- **Typography**: Unified font and text styling
- **Layout**: Responsive design with proper spacing
- **Icons**: Emoji-based visual indicators

### Interactive Elements
- **Sidebar Configuration**: Clean parameter controls
- **Progress Indicators**: Real-time simulation progress
- **Status Messages**: Clear feedback for user actions
- **Export Buttons**: Easy access to result downloads

## 🔧 Tab Development

### Adding New Tabs
```python
# Create new tab file
def render_new_tab():
    """Render a new custom tab."""
    st.header("New Tab")
    st.write("This is a new tab component")
    
    # Add your tab content here
    st.button("Custom Action")

# Add to main app
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Simulation", "Anchor Comparison", "New Tab"])

with tab4:
    render_new_tab()
```

### Tab Dependencies
```python
# Check dependencies before rendering
def render_dependent_tab():
    """Render tab that depends on simulation results."""
    if 'simulation_results' not in st.session_state:
        st.warning("Please run simulation first")
        return
    
    # Render tab content
    st.header("Dependent Tab")
    # ... tab content
```

## 🧪 Testing

Tab components have comprehensive test coverage:

- **Component Tests**: Individual tab functionality
- **Integration Tests**: Tab interaction and state management
- **UI Tests**: Visual rendering and user interaction
- **Error Handling**: Graceful error display

Run tab tests:
```bash
pytest tests/test_ui.py -v
pytest tests/test_ui_integration.py -v
```

## 📱 User Experience

### Accessibility Features
- **Clear Navigation**: Intuitive tab structure
- **Helpful Tooltips**: Context-sensitive information
- **Error Messages**: Clear feedback for issues
- **Export Options**: Multiple format support

### Performance Optimization
- **Lazy Loading**: Content loaded on demand
- **State Caching**: Results cached between tab switches
- **Memory Management**: Efficient data handling
- **Responsive Design**: Works on different screen sizes

## 📚 References

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **UMAP**: Dimensionality reduction algorithm
- **Material Design**: UI/UX design principles

## 📝 Recent Updates

### Interactive Temperature Exploration
- **Temperature slider integration**: Added interactive slider in anchor comparison tab for dynamic UMAP exploration
- **Disk-based snapshot loading**: Efficient loading of vector snapshots from disk based on temperature selection
- **Real-time UMAP updates**: Dynamic visualization updates as temperature slider moves
- **Memory optimization**: Temperature-based loading reduces memory usage for large simulations

### Enhanced Metrics Display
- **Three-column layout**: Critical Temperature, Cosine Distance, and Cosine Similarity displayed consistently
- **Improved Tc display**: Critical temperature prominently shown with enhanced help text
- **Pre-calculated metrics**: All temperature metrics calculated after simulation to avoid real-time computation
- **Consistent anchor comparison**: Both main metrics and interactive metrics use original anchor vectors
- **Disk-based snapshot support**: Proper handling of disk-based snapshots for recalculation

### Language Labeling Improvements
- **Fixed UMAP language display**: Language labels now show actual language codes (en, es, fr, etc.) instead of generic labels
- **Proper parameter passing**: Language codes correctly passed from concept files through simulation to UI
- **Consistent labeling**: All visualization components use consistent language codes

### Performance Enhancements
- **Efficient data retrieval**: Hash-based snapshot directory naming for unique simulation configurations
- **Smooth interaction**: Pre-calculated metrics enable responsive temperature slider interaction
- **Optimized memory usage**: Temperature-based snapshot loading instead of loading all snapshots at once

### Anchor Comparison Metrics Consistency
- **Consistent calculation method**: Both main metrics and interactive metrics now use original anchor vectors for comparison
- **Disk-based recalculation**: Fixed anchor comparison recalculation to properly handle disk-based snapshots when in-memory snapshots are not available
- **Session state synchronization**: All UI components display consistent metrics by updating session state after recalculation
- **Original anchor vector preservation**: Ensures comparison metrics reflect the relationship between original anchor language and evolved multilingual structure 