# Core Module Documentation

The `core/` directory contains the main simulation engine and scientific computation modules for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Modules](#modules)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The core modules implement the scientific foundation of the Semantic Ising Simulator, including:

- **Multilingual embedding processing** with caching and error recovery
- **Ising model simulation** with Metropolis and Glauber dynamics
- **Phase transition detection** using Binder cumulant method
- **Advanced comparison metrics** for anchor language analysis
- **Meta vector computation** using multiple aggregation methods
- **Temperature estimation** for optimal simulation ranges

## 📦 Modules

### 🔤 `embeddings.py`
**Purpose**: Multilingual embedding pipeline with caching and error recovery

**Key Functions**:
- `load_concept_embeddings(concept_name)` - Load multilingual translations
- `generate_embeddings(concept, encoder)` - Generate embeddings with caching
- `cache_embeddings(embeddings, concept, encoder)` - Store embeddings with validation

**Features**:
- Automatic caching to avoid recomputation
- Error recovery for corrupted cache files
- Support for multiple encoder models (LaBSE, SBERT, XLM-R)
- Normalized vector output

### 🎯 `anchor_config.py`
**Purpose**: Configurable anchor experiment design system

**Key Functions**:
- `configure_anchor_experiment(all_languages, anchor_language, include_anchor)` - Configure experiment design
- `validate_anchor_config(all_languages, anchor_language, include_anchor)` - Validate configuration
- `get_experiment_description(anchor_language, include_anchor, dynamics_languages)` - Generate descriptions

**Features**:
- Single-phase mode: anchor included in dynamics
- Two-phase mode: anchor compared to emergent structure
- Comprehensive validation and error handling

### 🌡️ `temperature_estimation.py`
**Purpose**: Intelligent temperature range estimation for optimal simulation

**Key Functions**:
- `estimate_critical_temperature(vectors)` - Estimate Tc from initial similarity
- `estimate_max_temperature(vectors)` - Estimate Tmax from energy fluctuations (uses 2.0× multiplier for conservative estimates)
- `estimate_practical_range(vectors, config_max_temperature=None)` - Estimate optimal [Tmin, Tmax] range with config respect
- `quick_scan_probe(vectors, T_range)` - Refine temperature range estimate

**Features**:
- Physics-based estimation using vector similarity and energy fluctuations
- Configurable max temperature parameter to respect user settings
- Conservative 2.0× multiplier for energy fluctuation scaling
- Automatic range validation and adjustment
- Integration with simulation workflow

### 🔥 `simulation.py`
**Purpose**: Core Ising model simulation engine

**Key Functions**:
- `run_temperature_sweep(vectors, T_range, store_all_temperatures=False)` - Main simulation driver
- `simulate_at_temperature(vectors, T)` - Single temperature simulation
- `update_vectors_ising(vectors, T, method)` - Ising update rules
- `collect_metrics(vectors, T)` - Comprehensive metric collection

**Update Methods**:
- **Metropolis**: Standard Metropolis-Hastings acceptance criterion
- **Glauber**: Heat-bath probability-based updates

**Features**:
- Multi-replica support for statistical averaging
- Memory management with configurable snapshot storage
- Convergence detection and error handling
- Comprehensive metric collection
- Temperature-dependent clustering thresholds (0.8-0.95 range)

### 📊 `dynamics.py`
**Purpose**: Correlation analysis and dynamic properties

**Key Functions**:
- `compute_correlation_matrix(vectors)` - Compute correlation matrix C_ij
- `compute_correlation_length(vectors, lang_dist_matrix=None)` - Compute correlation length ξ
- `alignment_curvature(alignment_curve, temperatures)` - Compute d²M/dT²

**Features**:
- Optional linguistic distance weighting
- Exponential decay fitting for correlation length
- Finite difference computation for curvature

### 🔍 `phase_detection.py`
**Purpose**: Critical temperature detection and phase transition analysis

**Key Functions**:
- `find_critical_temperature(metrics_dict)` - Detect Tc using the log(ξ) derivative (knee in correlation length vs temperature) as the default method. If correlation_length is not available, it falls back to the old methods.

**Features**:
- Robust Binder cumulant method for Tc detection
- Integration with clustering analysis

### 🧠 `meta_vector.py`
**Purpose**: Meta vector computation using multiple aggregation methods

**Key Functions**:
- `compute_meta_vector(vectors, method, weights=None)` - Main meta vector function
- `compute_centroid(vectors)` - Mean-based aggregation
- `compute_medoid(vectors)` - Closest-to-centroid vector
- `compute_geometric_median(vectors)` - Robust geometric median
- `compute_first_principal_component(vectors)` - PCA-based aggregation

**Methods**:
- **centroid**: Arithmetic mean of normalized vectors
- **medoid**: Vector closest to centroid
- **weighted_mean**: Weighted average with custom weights
- **geometric_median**: Robust median using Weiszfeld's algorithm
- **first_principal_component**: First principal component

### 🔗 `comparison_metrics.py`
**Purpose**: Advanced comparison metrics for anchor language analysis

**Key Functions**:
- `compare_anchor_to_multilingual(anchor_vectors, multilingual_vectors, tc, metrics)` - Anchor comparison at critical temperature
  - **Primary Metrics**: Cosine distance and cosine similarity (meaningful for single vector comparison)
  - **Set-based Metrics**: Procrustes, CKA, EMD, KL divergence (set to NaN for single vector comparison)
  - **Returns**: Dictionary with all metrics, but only cosine metrics are meaningful
- `compute_procrustes_distance(vectors_a, vectors_b)` - Structural alignment (requires multiple vectors)
- `compute_cka_similarity(vectors_a, vectors_b)` - Centered Kernel Alignment (requires multiple vectors)
- `compute_emd_distance(vectors_a, vectors_b)` - Earth Mover's Distance (requires multiple vectors)
- `compute_kl_divergence(vectors_a, vectors_b)` - KL divergence (requires multiple vectors)

**Primary Metric**:
- **Cosine Distance**: The community standard for semantic similarity in modern embeddings. Focuses on vector orientation rather than length, making it invariant to normalization, temperature scaling, and L2 regularization.

**Features**:
- Meta-vector comparison: Anchor compared to centroid of multilingual set
- Cosine distance as primary semantic metric
- Set-based metrics (Procrustes, CKA, EMD, KL) set to NaN for single vector comparison
- Automatic interpretation and scoring
- Integration with post-analysis workflow

### 📈 `post_analysis.py`
**Purpose**: Post-simulation analysis and result interpretation

**Key Functions**:
- `analyze_simulation_results(simulation_results, anchor_vectors, tc)` - Comprehensive analysis
- `generate_visualization_data(simulation_results, analysis_results)` - Prepare visualization data
- `interpret_analysis_results(analysis_results)` - Generate human-readable interpretations

**Features**:
- Anchor comparison at critical temperature
- Correlation analysis and interpretation
- Data preparation for UI visualizations
- Comprehensive error handling and validation

### 🔬 `physics.py`
**Purpose**: Energy calculations and physical consistency

**Key Functions**:
- `total_system_energy(vectors, J=1.0)` - Compute total system energy

**Features**:
- Consistent Hamiltonian with Metropolis updates
- Proper energy scaling with coupling strength
- Validation of physical consistency

### 🎯 `clustering.py`
**Purpose**: Vector clustering for power law analysis

**Key Functions**:
- `cluster_vectors(vectors, threshold=0.8, min_cluster_size=2)` - Cluster vectors by similarity

**Features**:
- Cosine similarity-based clustering
- Configurable threshold and minimum cluster size
- Integration with power law analysis

## 🔗 Dependencies

### Internal Dependencies
```
core/
├── embeddings.py (independent)
├── anchor_config.py (independent)
├── temperature_estimation.py (independent)
├── simulation.py (depends on dynamics.py, physics.py)
├── dynamics.py (independent)
├── phase_detection.py (depends on dynamics.py, clustering.py)
├── meta_vector.py (independent)
├── comparison_metrics.py (independent)
├── post_analysis.py (depends on comparison_metrics.py, phase_detection.py)
├── physics.py (independent)
└── clustering.py (independent)
```

### External Dependencies
- **NumPy**: Numerical computations and array operations
- **SciPy**: Scientific computing (curve fitting, statistics)
- **scikit-learn**: Machine learning (PCA, clustering)
- **Sentence Transformers**: Multilingual embeddings
- **PyTorch**: Deep learning backend (optional)

## 💡 Usage Examples

### Basic Simulation Workflow
```python
from core.embeddings import generate_embeddings
from core.simulation import run_temperature_sweep
from core.phase_detection import find_critical_temperature

# Generate embeddings
vectors, languages = generate_embeddings("dog", "LaBSE")

# Run temperature sweep
results = run_temperature_sweep(vectors, T_range=[0.1, 3.0])

# Detect critical temperature
tc = find_critical_temperature(results)
```

### Anchor Configuration
```python
from core.anchor_config import configure_anchor_experiment

# Configure two-phase experiment
dynamics_langs, comparison_langs = configure_anchor_experiment(
    all_languages=["en", "es", "fr", "de"],
    anchor_language="en",
    include_anchor=False
)
```

### Meta Vector Computation
```python
from core.meta_vector import compute_meta_vector

# Compute centroid meta vector
result = compute_meta_vector(vectors, method="centroid")
meta_vector = result['meta_vector']
```

### Post-Analysis
```python
from core.post_analysis import analyze_simulation_results

# Perform comprehensive analysis
analysis = analyze_simulation_results(
    simulation_results, 
    anchor_vectors, 
    tc
)
```

## 🧪 Testing

All core modules have comprehensive test coverage:

- **Unit tests**: Individual function testing
- **Integration tests**: End-to-end workflow testing
- **Validation tests**: Mathematical property verification
- **Performance tests**: Scalability and memory usage

Run tests with:
```bash
pytest tests/test_*.py -v
```

## 📚 References

- **Ising Model**: Statistical physics foundation
- **Binder Cumulant**: Critical temperature detection method
- **LaBSE**: Language-agnostic BERT sentence embeddings
- **UMAP**: Dimensionality reduction for visualization
- **Procrustes Analysis**: Structural alignment method 

## Phase Detection (core/phase_detection.py)

- **find_critical_temperature**: Now detects Tc using the log(ξ) derivative (knee in correlation length vs temperature) as the default method. This is more robust and physically meaningful than the previous alignment-based or Binder cumulant methods. If correlation_length is not available, it falls back to the old methods.

- **detect_powerlaw_regime**: Unchanged.

## Comparison Metrics (core/comparison_metrics.py)

- **compare_anchor_to_multilingual**: Now compares anchor vector to meta-vector of multilingual set (not individual vectors). Uses cosine distance as the primary semantic metric, with set-based metrics (Procrustes, CKA, EMD, KL) set to NaN for single vector comparison. This follows community standards for semantic similarity in modern embeddings. 

### 📝 Recent Updates
- Improved synchronization between backend Tc detection and UI display: the critical temperature (Tc) value is now always consistent between backend logic and user interface charts.
- Removed all debug output from core modules for a cleaner user experience.
- The convergence summary chart now only displays the vertical Tc line (critical temperature), with the convergence threshold line removed for clarity.