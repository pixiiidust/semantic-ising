-- Semantic Ising Simulator - Directory Structure (Updated)
-- Version: 1.0.0 (All Phases Complete)
-- Test Status: 242/242 tests passing

--[[ 
📋 Table of Contents
==================

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
   - [Configuration Management](#configuration-management)
   - [Core Simulation Engine](#core-simulation-engine)
   - [UI Components](#ui-components)
   - [Export and I/O](#export-and-io)
4. [Module Dependencies](#module-dependencies)
5. [Test Coverage](#test-coverage)

Use Ctrl+F and search for section headers (e.g. "## Core Components") to navigate
--]]

--[[ 
## Project Overview
==================
This directory structure defines the semantic-ising project architecture,
a multilingual semantic Ising model simulator that explores how semantically
identical words across languages potentially converge in their embedding space.

Project Status:
- Version: 1.0.0
- Test Coverage: 100% (242/242 tests)
- API Status: Locked (v0.2)
- Development Phase: Complete
--]]

## 📁 Project Structure

```
semantic-ising/
├── app.py                 # Main Streamlit application
├── main.py               # CLI interface
├── config/               # Configuration management
│   ├── defaults.yaml     # Default parameters and validation schema
│   ├── validator.py      # Configuration validation logic
│   └── config.md         # Configuration documentation
├── core/                 # Core simulation engine
│   ├── simulation.py     # Temperature sweeps & Ising updates
│   ├── embeddings.py     # Multilingual embedding pipeline
│   ├── phase_detection.py # Critical temperature detection
│   ├── post_analysis.py  # Post-simulation analysis
│   ├── dynamics.py       # Ising dynamics implementation
│   ├── physics.py        # Energy calculations
│   ├── clustering.py     # Vector clustering
│   ├── meta_vector.py    # Meta vector computation
│   ├── comparison_metrics.py # Similarity metrics
│   ├── anchor_config.py  # Anchor language handling
│   ├── temperature_estimation.py # Temperature range estimation
│   └── core.md          # Core module documentation
├── ui/                   # User interface components
│   ├── charts.py         # Interactive visualizations
│   ├── components.py     # Reusable UI components
│   ├── tabs/             # Tab-specific components
│   │   ├── overview.py   # Overview tab implementation
│   │   ├── simulation.py # Simulation control tab
│   │   ├── anchor_comparison.py # Anchor analysis tab
│   │   └── tabs.md      # Tab documentation
│   └── ui.md            # UI module documentation
├── data/                 # Data storage
│   ├── concepts/         # Multilingual concept files
│   │   ├── *_25.json    # 25-language concept files
│   │   ├── *_75.json    # 75-language concept files
│   │   └── concepts.md  # Concept file documentation
│   ├── embeddings/       # Cached embeddings
│   │   └── embeddings.md # Embedding cache documentation
│   └── snapshots/        # Simulation vector snapshots
│       ├── {concept}_{hash}/ # Hash-based directory naming
│       │   ├── snapshot_T*.pkl # Temperature-indexed snapshots
│       │   └── metadata.json   # Simulation metadata
│       └── data.md       # Data module documentation
├── tests/                # Test suite (242 tests)
│   ├── fixtures/        # Test fixtures and data
│   ├── test_*.py       # Test implementation files
│   └── tests.md        # Test suite documentation
├── export/              # Export and I/O utilities
│   ├── logger.py       # Structured logging system
│   ├── io.py          # File I/O operations
│   ├── results.py     # Results export handling
│   ├── ui_helpers.py  # UI-specific export utilities
│   └── export.md      # Export module documentation
├── docs/               # Project documentation
│   ├── SETUP.md       # Detailed setup guide
│   └── CONTRIBUTING.md # Contribution guidelines
├── README.md           # Project overview
├── requirements.txt    # Python dependencies
└── directory_structure.lua  # This canonical structure file
```

--[[ 
## File Categories
=================
1. Core Implementation (.py):
   - Main application files
   - Core simulation modules
   - Analysis components
   - UI implementation

2. Configuration Files:
   - YAML configurations
   - JSON data files
   - Requirements specification

3. Documentation (.md):
   - Module documentation
   - Setup guides
   - Usage instructions

4. Test Files:
   - Unit tests
   - Integration tests
   - Test fixtures
   - Performance tests
--]]

--[[ 
## Core Components
=================
The project is organized into four main components:
1. Configuration Management: Parameter validation and defaults
2. Core Simulation Engine: Ising dynamics and analysis
3. UI Components: Interactive visualization and control
4. Export and I/O: Data persistence and logging
--]]

--[[ 
## Module Dependencies
=====================
Key dependency relationships between components:
1. UI depends on Core Simulation
2. Core Simulation depends on Configuration
3. Export depends on all components
4. All components use logging utilities
--]]

--[[ 
## Test Coverage
===============
- Total Tests: 242
- Coverage: 100%
- Key Test Areas:
  * Core Simulation (89 tests)
  * UI Integration (45 tests)
  * Configuration (31 tests)
  * Export/IO (28 tests)
  * Performance (30 tests)
  * Post-Analysis (19 tests)
--]]

--[[ 
## Recent Updates (Latest Version)
==================================
Latest enhancements and improvements:

1. **Disk-based Snapshot Storage**:
   - Added persistent storage of simulation vectors at each temperature step
   - Hash-based directory naming for unique simulation configurations
   - Temperature-indexed files for efficient retrieval
   - Language code preservation in snapshots

2. **Interactive UMAP Visualization**:
   - Temperature slider for dynamic exploration of semantic structure
   - UMAP zoom control with 2.0× zoom factor
   - Real-time vector loading from disk snapshots
   - Language code preservation in UMAP plots

3. **Enhanced Metrics Display**:
   - Three-column layout: Critical Temperature, Cosine Distance, Cosine Similarity
   - Pre-calculated metrics for all temperatures
   - Improved Tc display with enhanced help text
   - Debug output cleanup for cleaner UI

4. **Performance Optimizations**:
   - Memory efficiency through temperature-based snapshot loading
   - Real-time responsiveness with pre-calculated metrics
   - Efficient data retrieval with hash-based directory naming
--]]

return {
  -- Configuration Management
  ["config/"] = {
    modules = {
      ["validator.py"] = {
        functions = {"validate_config", "load_config"},
        purpose = "Configuration validation and loading",
        produces = {"validated_config", "loaded_config"}
      },
      ["defaults.yaml"] = {
        purpose = "Default configuration with validation schema",
        produces = {"default_config"}
      }
    },
    status = "COMPLETE"
  },

  -- Core Simulation Engine
  ["core/"] = {
    modules = {
      ["simulation.py"] = {
        functions = {
          "run_temperature_sweep",  -- Main driver with multi-replica support and language parameter
          "simulate_at_temperature",  -- Ising update logic
          "update_vectors_ising",  -- Update rule dispatcher
          "update_vectors_metropolis",  -- Metropolis updates
          "update_vectors_glauber",  -- Heat-bath updates
          "collect_metrics",  -- Metric collection
          "compute_alignment",  -- Alignment calculation
          "compute_entropy",  -- Entropy calculation
          "_save_snapshot_to_disk",  -- Disk-based snapshot storage
          "_load_snapshot_from_disk",  -- Load snapshots from disk
          "_get_available_snapshot_temperatures"  -- Get available snapshot temperatures
        },
        depends_on = {"dynamics.py", "physics.py"},
        produces = {"temperature_metrics", "vector_snapshots", "snapshot_directory"}
      },
      
      ["embeddings.py"] = {
        functions = {
          "load_concept_embeddings",  -- Load and validate translations
          "generate_embeddings",  -- Generate with caching
          "cache_embeddings"  -- Store with validation
        },
        produces = {"embeddings_array", "language_list", "cache_filepath"}
      },

      ["dynamics.py"] = {
        functions = {
          "compute_correlation_matrix",  -- Cosine similarity matrix
          "compute_correlation_length",  -- With linguistic weighting
          "alignment_curvature"  -- For Tc detection
        },
        produces = {"correlation_matrix", "correlation_length", "curvature_array"}
      },

      ["physics.py"] = {
        functions = {"total_system_energy"},
        purpose = "Hamiltonian energy calculation",
        produces = {"energy_value"}
      },

      ["phase_detection.py"] = {
        functions = {"find_critical_temperature"},
        purpose = "Tc detection via log(ξ) derivative",
        depends_on = {"dynamics.py", "clustering.py"},
        produces = {"critical_temperature"}
      },

      ["clustering.py"] = {
        functions = {"cluster_vectors"},
        purpose = "Cosine similarity clustering",
        produces = {"cluster_list"}
      },

      ["meta_vector.py"] = {
        functions = {
          "compute_meta_vector",  -- Main interface
          "compute_centroid",  -- Mean of normalized vectors
          "compute_medoid",  -- Closest to centroid
          "compute_weighted_mean",  -- Weighted average
          "compute_geometric_median",  -- Weiszfeld algorithm
          "compute_first_principal_component"  -- PCA-based
        },
        produces = {"meta_vector_result"}
      },

      ["comparison_metrics.py"] = {
        functions = {
          "compute_procrustes_distance",
          "compute_cka_similarity",
          "compute_emd_distance",
          "compute_kl_divergence",
          "compare_anchor_to_multilingual"
        },
        produces = {"comparison_metrics"}
      },

      ["temperature_estimation.py"] = {
        functions = {
          "estimate_critical_temperature",  -- Initial Tc estimate
          "estimate_max_temperature",  -- Conservative max T
          "estimate_practical_range",  -- Range with validation
          "quick_scan_probe",  -- Range refinement
          "validate_temperature_range"  -- Range validation
        },
        depends_on = {"simulation.py"},
        produces = {"estimated_tc", "estimated_tmax", "refined_range"}
      }
    },
    status = "COMPLETE"
  },

  -- UI Components
  ["ui/"] = {
    modules = {
      ["charts.py"] = {
        purpose = "Interactive visualizations with UMAP zoom control and temperature slider integration",
        functions = {
          "plot_entropy_vs_temperature",  -- Entropy vs temperature plot
          "plot_full_umap_projection",  -- UMAP projection with zoom control
          "plot_correlation_decay",  -- Correlation decay plot
          "plot_correlation_length_vs_temperature",  -- Correlation length vs temperature
          "plot_alignment_vs_temperature",  -- Alignment vs temperature
          "plot_energy_vs_temperature"  -- Energy vs temperature
        },
        depends_on = {"core/simulation.py", "core/phase_detection.py"}
      },
      ["components.py"] = {
        purpose = "Reusable UI elements",
        depends_on = {"core/temperature_estimation.py"}
      },
      ["tabs/"] = {
        purpose = "Tab-specific interfaces",
        depends_on = {"charts.py", "components.py"}
      }
    },
    status = "COMPLETE"
  },

  -- Export and I/O
  ["export/"] = {
    modules = {
      ["logger.py"] = {
        functions = {"init_logger", "log_event", "log_exception"},
        purpose = "Structured logging with error tracking",
        produces = {"logger_instance", "log_entries"}
      }
    },
    status = "COMPLETE"
  },

  -- Data Storage
  ["data/"] = {
    modules = {
      ["concepts/"] = {
        purpose = "Multilingual concept translation files",
        produces = {"concept_translations", "language_codes"}
      },
      ["embeddings/"] = {
        purpose = "Cached embedding arrays",
        produces = {"embedding_cache", "cached_vectors"}
      },
      ["snapshots/"] = {
        purpose = "Simulation vector snapshots with hash-based directory naming",
        functions = {
          "snapshot_T*.pkl",  -- Temperature-indexed snapshot files
          "metadata.json"     -- Simulation metadata storage
        },
        produces = {"vector_snapshots", "simulation_metadata", "temperature_index"}
      }
    },
    status = "COMPLETE"
  }
}