# 🛠️ Detailed Setup Guide

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [Implementation Details](#implementation-details)
- [Data Management](#data-management)
- [Running Experiments](#running-experiments)
- [Output & Analysis](#output--analysis)

## 📖 Overview
The Semantic Ising Simulator explores how semantically identical words across languages converge in their embedding space. It uses statistical physics concepts (Ising model dynamics) to detect phase transitions in multilingual semantic alignments.

### Key Features
- Multilingual semantic analysis (25-75 languages)
- Phase transition detection in semantic space
- Real-time visualization and monitoring
- Comprehensive metric analysis
- Anchor language comparisons

## 💻 System Requirements

### Hardware
- CPU: Multi-core recommended for parallel simulations
- RAM: 4GB+ for standard simulations (8GB+ for 70+ languages)
- GPU: CUDA-capable (optional, for faster embeddings)
- Storage: 500MB+ for embeddings cache

### Software
- Python 3.10 or 3.11 (recommended)
- Git for version control
- CUDA Toolkit 11.8+ (if using GPU)

## ⚡ Quick Start

### Basic Installation
```bash
# Clone and setup
git clone https://github.com/pixiiidust/semantic-ising.git
cd semantic-ising
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Run the simulator
streamlit run app.py
```

### Reproducing Results
1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Download concept files from `data/concepts/`
4. Run simulations with default parameters
5. Compare outputs with reference results in `docs/reference_results/`

## 🏗️ Project Architecture

The complete and up-to-date project structure is maintained in [directory_structure.lua](directory_structure.lua), which serves as the canonical source of truth for the project's architecture. Please refer to it for:

- Complete module hierarchy and dependencies
- File relationships and data flow
- Test coverage information
- Component documentation

### Core Components
1. **Simulation Engine** (`core/`):
   - Implements Metropolis-Hastings and Glauber dynamics
   - Manages temperature sweeps and convergence
   - Detects phase transitions using log(ξ) derivative method
   - Computes meta vectors and correlations
   - **Disk-based snapshot storage** for memory efficiency and large simulations

2. **Analysis Pipeline** (`core/`):
   - Generates and caches LaBSE embeddings
   - Analyzes correlations and clustering
   - Computes comparison metrics
   - Performs post-simulation analysis
   - **Language code preservation** in snapshots for proper visualization

3. **User Interface** (`ui/`):
   - Real-time visualization of simulations
   - Interactive parameter configuration
   - Progress monitoring and logging
   - Results analysis and export
   - **Interactive temperature slider** for dynamic UMAP exploration
   - **Enhanced metrics display** with three-column layout

## 🔧 Implementation Details

### Core Simulation
1. **Ising Dynamics**:
   ```python
   # core/simulation.py
   def update_vectors_metropolis(vectors, temperature):
       """Metropolis-Hastings update for semantic vectors.
       
       Args:
           vectors (np.ndarray): Shape (n_langs, embedding_dim)
           temperature (float): Current temperature
           
       Returns:
           np.ndarray: Updated vectors
       """
   ```

2. **Phase Detection**:
   ```python
   # core/phase_detection.py
   def find_critical_temperature(correlation_lengths, temperatures):
       """Detect Tc using log(ξ) derivative method.
       
       Returns:
           float: Critical temperature
       """
   ```

### Embedding Pipeline
1. **LaBSE Model**:
   - Architecture: BERT-based
   - Input: UTF-8 text
   - Output: 768-dimensional vectors
   - Language Support: 109 languages
   - Paper: [LaBSE: Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852)

2. **Caching System**:
   ```python
   # core/embeddings.py
   def cache_embeddings(vectors, languages, concept):
       """Cache embeddings with validation."""
   ```

## 📊 Data Management

### Input Data
1. **Concept Files** (`data/concepts/`):
   ```json
   {
     "en": "dog",
     "es": "perro",
     "fr": "chien"
   }
   ```
   - Format: UTF-8 JSON
   - Naming: `{concept}_translations_{25|75}.json`
   - Validation: Language code check (ISO 639-1)

2. **Embeddings** (`data/embeddings/`):
   - Format: NumPy arrays (.npy)
   - Dimensions: (n_languages, 768)
   - Naming: `{concept}_{timestamp}.npy`
   - Auto-generated and cached

3. **Simulation Snapshots** (`data/snapshots/`):
   - Format: Pickle files (.pkl) containing numpy arrays and metadata
   - Naming: `snapshot_T{temperature}.pkl` for temperature-based indexing
   - Content: Vector snapshots, language codes, and simulation metadata
   - **Hash-based directory naming** for unique simulation configurations
   - **Automatic snapshot indexing** for efficient temperature-based retrieval

### Dependencies
Core requirements (see `requirements.txt`):
```
sentence-transformers>=2.2.0  # LaBSE embeddings
torch>=2.0.0                 # Neural network operations
umap-learn>=0.5.3           # Dimensionality reduction
plotly>=5.13.0              # Interactive visualization
streamlit>=1.22.0           # Web interface
numpy>=1.23.5               # Numerical computation
scipy>=1.10.0               # Scientific computing
pyyaml>=6.0                 # Configuration
```

## 🆕 Latest Features

### Interactive UMAP Visualization
- **Temperature slider integration**: Dynamic exploration of semantic structure across temperature steps
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

## 🔬 Running Experiments

### Configuration
1. **Basic Setup**:
   - Select concept and language set
   - Configure anchor language (optional)
   - Set/auto-estimate temperature range

2. **Advanced Parameters**:
   - Temperature: 0.1 to 5.0 (auto-estimated)
   - Update Rules: Metropolis-Hastings or Glauber
   - Convergence Threshold: ΔS < 1e-6
   - Early Stop: 1000 steps without convergence

### Monitoring
- Real-time progress tracking
- Live metric updates
- Interactive visualizations
- Error logging and recovery

## 📈 Output & Analysis

### Metrics & Measurements
1. **Correlation Length (ξ)**:
   - Definition: ξ = √(-1/log(C(r)))
   - Usage: Critical temperature detection
   - Implementation: `core/dynamics.py:compute_correlation_length()`

2. **Convergence**:
   - Primary: Entropy stabilization (ΔS)
   - Secondary: Energy fluctuations
   - Threshold: ΔS < 1e-6 over 100 steps
   - Code: `core/simulation.py:check_convergence()`

3. **Similarity Metrics**:
   - Basic: Cosine similarity/distance
   - Advanced: Procrustes, CKA, EMD, KL
   - Implementation: `core/comparison_metrics.py`

### Output Files
1. **Simulation Data**:
   - Temperature sweep results (.csv)
   - Vector snapshots at Tc (.npy)
   - Convergence metrics (.json)
   - UMAP projections (.html)

2. **Analysis Results**:
   - Phase transition plots
   - Correlation analysis
   - Anchor comparisons
   - Performance metrics

### Visualization
- Interactive UMAP projections with temperature slider
- Temperature sweep plots
- Correlation decay curves
- Convergence diagnostics

For detailed implementation references, see inline documentation in respective modules.

# Python Virtual Environment Setup (.venv)

Using a virtual environment is highly recommended for Python projects to avoid dependency conflicts and keep your global Python installation clean.

### 1. Create a Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv .venv
```

#### macOS/Linux:
```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

#### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

#### macOS/Linux:
```bash
source .venv/bin/activate
```

You should see your prompt change to indicate the environment is active, e.g.:
```
(.venv) D:\semantic-ising>
```

### 3. Install Project Dependencies

With the virtual environment activated, run:
```bash
pip install -r requirements.txt
```

### 4. Deactivate the Virtual Environment

When finished, you can deactivate with:
```bash
deactivate
```

### 5. Common Issues & Troubleshooting

- **'python' not found:**
  - Try `python3` instead of `python` on macOS/Linux.
- **Activation script not found:**
  - Ensure you are in the project root and `.venv` exists.
- **Permission denied (macOS/Linux):**
  - Run `chmod +x .venv/bin/activate` if needed.
- **PowerShell execution policy (Windows):**
  - If you see a policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Dependencies not installing:**
  - Ensure your virtual environment is activated before running `pip install`.

## 6. Additional Resources
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip documentation](https://pip.pypa.io/en/stable/)