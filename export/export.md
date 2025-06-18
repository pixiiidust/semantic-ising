# Export Module Documentation

The `export/` directory contains comprehensive export and I/O utilities for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Files](#files)
- [Core Functions](#core-functions)
- [Export Formats](#export-formats)
- [UI Integration](#ui-integration)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The export module provides robust data export capabilities for the Semantic Ising Simulator, featuring:

- **Multiple export formats** (JSON, CSV, NumPy, PNG)
- **Comprehensive result archiving** with metadata
- **UI integration** for Streamlit downloads
- **Error handling** and validation
- **Cross-platform compatibility**

## 📁 Files

### 💾 `io.py`
**Purpose**: Basic I/O operations with error handling

**Key Functions**:
- `save_json(obj, filepath)` - Store objects as JSON
- `save_csv(data, filepath)` - Store data as CSV
- `save_embeddings(vectors, filename, filetype, output_dir)` - Save embeddings in multiple formats

**Features**:
- UTF-8 encoding support
- Comprehensive error handling
- Cross-platform path handling
- Input validation

### 📊 `results.py`
**Purpose**: Comprehensive simulation results export

**Key Functions**:
- `export_results(metrics, tc, meta_result, concept, output_dir, comparison_metrics, experiment_config)` - Export all results

**Features**:
- Multi-format export (CSV, JSON, NumPy)
- Metadata and summary generation
- Anchor comparison reports
- Experiment configuration tracking

### 🖥️ `ui_helpers.py`
**Purpose**: UI integration for Streamlit downloads

**Key Functions**:
- `export_csv_results(simulation_results, analysis_results)` - Export results as CSV
- `export_vectors_at_tc(simulation_results)` - Export vectors at critical temperature
- `export_charts(simulation_results, analysis_results)` - Export charts as PNG
- `export_config_file()` - Export current configuration

**Features**:
- Temporary file creation for downloads
- Automatic cleanup handling
- Graceful fallbacks for missing dependencies
- Streamlit integration

### 📄 `export.md`
**Purpose**: This documentation file

## 🔧 Core Functions

### JSON Export
```python
from export.io import save_json

# Save simulation results
results = {
    'critical_temperature': 1.5,
    'alignment_curve': [0.8, 0.7, 0.6],
    'languages': ['en', 'es', 'fr']
}

save_json(results, 'output/simulation_results.json')
```

### CSV Export
```python
from export.io import save_csv

# Save metrics data
metrics = {
    'temperatures': [0.1, 0.5, 1.0],
    'alignment': [0.9, 0.8, 0.7],
    'entropy': [0.1, 0.2, 0.3]
}

save_csv(metrics, 'output/metrics.csv')
```

### Embedding Export
```python
from export.io import save_embeddings

# Save meta vector
meta_vector = np.random.randn(768)
save_embeddings(
    vectors=meta_vector,
    filename='dog_meta',
    filetype='meta',
    output_dir='output/'
)
```

### Comprehensive Results Export
```python
from export.results import export_results

# Export complete simulation results
export_results(
    metrics=simulation_metrics,
    tc=critical_temperature,
    meta_result=meta_vector_result,
    concept='dog',
    output_dir='output/',
    comparison_metrics=anchor_comparison,
    experiment_config=experiment_config
)
```

## 📄 Export Formats

### JSON Format
```json
{
  "concept": "dog",
  "critical_temperature": 1.523,
  "meta_vector_method": "centroid",
  "simulation_params": {
    "temperature_range": [0.1, 3.0],
    "n_temperatures": 50
  },
  "anchor_comparison": {
    "procrustes_distance": 0.234,
    "cka_similarity": 0.876,
    "emd_distance": 0.123,
    "cosine_similarity": 0.789
  },
  "experiment_config": {
    "anchor_language": "en",
    "include_anchor": false,
    "dynamics_languages": ["es", "fr", "de", "it"]
  }
}
```

### CSV Format
```csv
temperatures,alignment,entropy,energy,correlation_length
0.1,0.923,0.077,-0.923,2.456
0.5,0.856,0.144,-0.856,1.789
1.0,0.734,0.266,-0.734,1.234
1.5,0.612,0.388,-0.612,0.987
2.0,0.489,0.511,-0.489,0.654
```

### NumPy Format
- **File extension**: `.npy`
- **Data type**: `float32`
- **Shape**: `(n_languages, embedding_dimension)`
- **Normalization**: Unit-length vectors

### PNG Format
- **Chart exports**: Interactive Plotly charts as static images
- **Resolution**: High-quality for publications
- **Fallback**: Text file if kaleido not available

## 🖥️ UI Integration

### Streamlit Download Integration
```python
import streamlit as st
from export.ui_helpers import export_csv_results

# Export button in UI
if st.button("📄 Export Results"):
    csv_path = export_csv_results(simulation_results, analysis_results)
    
    with open(csv_path, 'rb') as f:
        st.download_button(
            label="Download CSV",
            data=f.read(),
            file_name="simulation_results.csv",
            mime="text/csv"
        )
```

### Vector Export
```python
from export.ui_helpers import export_vectors_at_tc

# Export vectors at critical temperature
if 'vector_snapshots' in simulation_results:
    npy_path = export_vectors_at_tc(simulation_results)
    
    with open(npy_path, 'rb') as f:
        st.download_button(
            label="Download Vectors",
            data=f.read(),
            file_name="vectors_at_tc.npy",
            mime="application/octet-stream"
        )
```

### Chart Export
```python
from export.ui_helpers import export_charts

# Export charts as PNG
png_path = export_charts(simulation_results, analysis_results)

with open(png_path, 'rb') as f:
    st.download_button(
        label="Download Charts",
        data=f.read(),
        file_name="simulation_charts.png",
        mime="image/png"
    )
```

## 💡 Usage Examples

### Basic Export Workflow
```python
from export.results import export_results
from core.simulation import run_temperature_sweep
from core.phase_detection import find_critical_temperature

# Run simulation
results = run_temperature_sweep(vectors, T_range=[0.1, 3.0])

# Detect critical temperature
tc = find_critical_temperature(results)

# Export results
export_results(
    metrics=results,
    tc=tc,
    meta_result=meta_vector_result,
    concept='dog',
    output_dir='./results/'
)
```

### Custom Export Configuration
```python
from export.io import save_json, save_csv

# Custom export with specific formats
def custom_export(simulation_results, output_dir):
    # Export metrics as CSV
    save_csv(simulation_results['metrics'], f'{output_dir}/metrics.csv')
    
    # Export summary as JSON
    summary = {
        'concept': 'dog',
        'critical_temperature': simulation_results.get('tc'),
        'n_languages': len(simulation_results.get('languages', [])),
        'temperature_range': simulation_results['metrics']['temperatures'][[0, -1]].tolist()
    }
    save_json(summary, f'{output_dir}/summary.json')
```

### Batch Export
```python
import os
from export.results import export_results

# Export multiple concepts
concepts = ['dog', 'tree', 'house']
output_base = './results/'

for concept in concepts:
    # Run simulation for each concept
    results = run_simulation(concept)
    
    # Create concept-specific output directory
    output_dir = os.path.join(output_base, concept)
    os.makedirs(output_dir, exist_ok=True)
    
    # Export results
    export_results(
        metrics=results['metrics'],
        tc=results['tc'],
        meta_result=results['meta_vector'],
        concept=concept,
        output_dir=output_dir
    )
```

## 🔧 Error Handling

### Robust Error Recovery
- **Invalid paths**: Graceful handling with clear error messages
- **Missing dependencies**: Fallback behavior for optional packages
- **Data validation**: Input validation for all export functions
- **Cross-platform**: Windows-compatible path handling
- **Memory management**: Efficient temporary file handling

### Error Types Handled
```python
# File system errors
IOError: "Failed to save JSON to /invalid/path/file.json: [Errno 2] No such file or directory"

# Invalid input data
ValueError: "Vectors must be 2D array"

# Missing dependencies
ImportError: "pandas not available, skipping parquet export"

# Missing required data
KeyError: "Missing required key: 'temperatures'"
```

## 📊 Performance Characteristics

### Export Speed
- **JSON Export**: ~1ms for typical results
- **CSV Export**: ~5ms for 1000 data points
- **Embedding Export**: ~10ms for 768-dimensional vectors
- **Chart Export**: ~100ms (including Plotly rendering)

### Memory Usage
- **Temporary files**: Automatic cleanup after download
- **Large datasets**: Streaming export for large files
- **Vector snapshots**: Efficient memory management

## 🧪 Testing

Export functionality has comprehensive test coverage:

- **Unit tests**: Individual export function testing
- **Integration tests**: End-to-end export workflows
- **Error tests**: Invalid input and error condition handling
- **UI tests**: Streamlit integration testing

Run export tests:
```bash
pytest tests/test_export.py -v
```

## 📚 References

- **JSON**: Data interchange format
- **CSV**: Comma-separated values format
- **NumPy**: Numerical array storage
- **Plotly**: Interactive visualization library
- **Streamlit**: Web application framework 