-- Semantic Ising Simulator - Directory Structure (Updated)
-- Phase 1 Complete: Foundation & CLI Interface (API locked v0.2)
-- Phase 2 Complete: Multilingual Embedding Pipeline
-- Phase 2.5 Complete: Anchor Configuration System
-- Phase 2.8 Complete: Temperature Estimation System
-- Phase 3 Complete: Core Simulation Engine
-- Phase 4 Complete: Semantic Phase Transition Metrics
-- Phase 4.7 Complete: Advanced Comparison Metrics
-- Phase 5 Complete: Meta Vector Inference
-- Phase 6 Complete: Testing & Validation (242/242 tests passing)
-- Phase 7 Complete: Post-Simulation Analysis (242/242 tests passing)
-- Phase 8 Complete: Export & I/O (242/242 tests passing)
-- Phase 9 Complete: UI Updates with Anchor Configuration (COMPLETE)
-- All critical gaps have been addressed with canonical function versions

-- Table of Contents:
-- 1. Configuration Management (Phase 1)
-- 2. Export and Logging (Phase 1)
-- 3. Main CLI Interface (Phase 1)
-- 4. Test Suite (Phase 1)
-- 5. Embedding Handling (Phase 2)
-- 6. Anchor Configuration (Phase 2.5)
-- 7. Temperature Estimation (Phase 2.8)
-- 8. Core Simulation Engine (Phase 3)
-- 9. Dynamics and Correlation Analysis (Phase 4)
-- 10. Phase Detection and Clustering (Phase 4)
-- 11. Meta Vector Computation (Phase 5)
-- 12. Comparison Metrics (Phase 4.7)
-- 13. Test Files (Phase 4-6)
-- 14. Post-Simulation Analysis (Phase 7)
-- 15. Export and I/O (Phase 8)
-- 16. UI Components (Phase 9 - COMPLETE)

return {
  -- 1. Configuration management (Phase 1 - API locked v0.2)
  ["config/validator.py"] = {
    function = "validate_config, load_config",
    purpose = "Validate config parameters and provide defaults (API locked v0.2)",
    depends_on = {},
    produces = {"validated_config", "loaded_config"},
    status = "COMPLETE"
  },
  
  ["config/defaults.yaml"] = {
    function = "Default configuration file",
    purpose = "Store config with validation schema including anchor configuration (API locked v0.2)",
    depends_on = {},
    produces = {"default_config"},
    status = "COMPLETE"
  },
  
  ["config/config.md"] = {
    function = "Configuration documentation",
    purpose = "Document configuration module structure and usage",
    depends_on = {},
    produces = {"documentation"},
    status = "COMPLETE"
  },
  
  -- 2. Export and logging (Phase 1 - API locked v0.2)
  ["export/logger.py"] = {
    function = "init_logger, log_event, log_exception",
    purpose = "Structured logs with error tracking (API locked v0.2)",
    depends_on = {},
    produces = {"logger_instance", "log_entries"},
    status = "COMPLETE"
  },
  
  ["export/export.md"] = {
    function = "Export documentation",
    purpose = "Document export module structure and logging utilities",
    depends_on = {},
    produces = {"documentation"},
    status = "COMPLETE"
  },
  
  -- 3. Main CLI interface (Phase 1 - API locked v0.2)
  ["main.py"] = {
    function = "parse_args, run_cli, run_simulation_from_file, main",
    purpose = "Top-level CLI entry with proper error handling (API locked v0.2)",
    depends_on = {"config/validator.py", "export/logger.py"},
    produces = {"cli_interface", "simulation_results"},
    status = "COMPLETE"
  },
  
  -- 4. Test suite (Phase 1 - API locked v0.2)
  ["tests/test_config_validation.py"] = {
    function = "TestConfigValidation",
    purpose = "Test config validation functionality (10 tests)",
    depends_on = {"config/validator.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_logging.py"] = {
    function = "TestLogging",
    purpose = "Test logging utilities (7 tests)",
    depends_on = {"export/logger.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_cli.py"] = {
    function = "TestCLI",
    purpose = "Test CLI interface functionality (10 tests)",
    depends_on = {"main.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/tests.md"] = {
    function = "Test documentation",
    purpose = "Document test module structure and coverage",
    depends_on = {},
    produces = {"documentation"},
    status = "COMPLETE"
  },
  
  -- 5. Embedding handling (Phase 2 - Complete)
  ["core/embeddings.py"] = {
    function = "load_concept_embeddings",
    purpose = "Read multilingual terms with error handling",
    depends_on = {},
    produces = {"translations_dict"},
    status = "COMPLETE"
  },
  
  ["core/embeddings.py"] = {
    function = "generate_embeddings",
    purpose = "Generate embeddings with caching and error recovery",
    depends_on = {"core/embeddings.py"},
    produces = {"embeddings_array", "language_list"},
    status = "COMPLETE"
  },
  
  ["core/embeddings.py"] = {
    function = "cache_embeddings",
    purpose = "Store embeddings with validation",
    depends_on = {},
    produces = {"cache_filepath"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 2 - Complete)
  ["tests/test_embeddings.py"] = {
    function = "TestEmbeddings",
    purpose = "Test embedding functionality (12 tests)",
    depends_on = {"core/embeddings.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- 6. Anchor configuration (Phase 2.5 - Complete)
  ["core/anchor_config.py"] = {
    function = "configure_anchor_experiment",
    purpose = "Configure which languages participate in dynamics vs comparison",
    depends_on = {},
    produces = {"dynamics_languages", "comparison_languages"},
    status = "COMPLETE"
  },
  
  ["core/anchor_config.py"] = {
    function = "validate_anchor_config",
    purpose = "Validate anchor configuration parameters",
    depends_on = {},
    produces = {"validation_result"},
    status = "COMPLETE"
  },
  
  ["core/anchor_config.py"] = {
    function = "get_experiment_description",
    purpose = "Generate human-readable experiment description",
    depends_on = {},
    produces = {"description_string"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 2.5 - Complete)
  ["tests/test_anchor_config.py"] = {
    function = "TestAnchorConfig",
    purpose = "Test anchor configuration functionality (12 tests)",
    depends_on = {"core/anchor_config.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- 7. Temperature estimation (Phase 2.8 - COMPLETE)
  ["core/temperature_estimation.py"] = {
    function = "estimate_critical_temperature",
    purpose = "Estimate critical temperature from initial vector similarity",
    depends_on = {},
    produces = {"estimated_tc"},
    status = "COMPLETE"
  },
  
  ["core/temperature_estimation.py"] = {
    function = "estimate_max_temperature",
    purpose = "Estimate maximum temperature from local field energy fluctuations",
    depends_on = {},
    produces = {"estimated_tmax"},
    status = "COMPLETE"
  },
  
  ["core/temperature_estimation.py"] = {
    function = "estimate_practical_range",
    purpose = "Estimate practical temperature range with padding and validation",
    depends_on = {"core/temperature_estimation.py"},
    produces = {"tmin", "tmax"},
    status = "COMPLETE"
  },
  
  ["core/temperature_estimation.py"] = {
    function = "quick_scan_probe",
    purpose = "Quick scan probe to refine temperature range estimate",
    depends_on = {"core/simulation.py"},
    produces = {"refined_range"},
    status = "COMPLETE"
  },
  
  ["core/temperature_estimation.py"] = {
    function = "validate_temperature_range",
    purpose = "Validate temperature range for simulation suitability",
    depends_on = {},
    produces = {"validation_result"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 2.8 - COMPLETE)
  ["tests/test_temperature_estimation.py"] = {
    function = "TestTemperatureEstimation",
    purpose = "Test temperature estimation functionality (26 tests)",
    depends_on = {"core/temperature_estimation.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- 8. Core simulation engine (Phase 3 - Complete)
  ["core/simulation.py"] = {
    function = "CANONICAL run_temperature_sweep",
    purpose = "Main temperature sweep driver with multi-replica support, memory management, and error handling",
    depends_on = {"core/dynamics.py", "core/physics.py"},
    produces = {"temperature_metrics", "vector_snapshots"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "CANONICAL simulate_at_temperature", 
    purpose = "Apply Ising-style update logic with convergence and vector return",
    depends_on = {"core/dynamics.py"},
    produces = {"metrics", "updated_vectors"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "update_vectors_ising",
    purpose = "Update vectors using specified Ising update rule (Metropolis/Glauber)",
    depends_on = {},
    produces = {"updated_vectors"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "update_vectors_metropolis",
    purpose = "Metropolis update rule for semantic Ising model",
    depends_on = {},
    produces = {"updated_vectors"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "update_vectors_glauber", 
    purpose = "Glauber (heat-bath) update rule for semantic Ising model",
    depends_on = {},
    produces = {"updated_vectors"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "collect_metrics",
    purpose = "Collect all metrics for current vector state",
    depends_on = {"core/dynamics.py", "core/physics.py"},
    produces = {"metrics_dict"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "compute_alignment",
    purpose = "Compute alignment metric: average absolute cosine similarity between all vector pairs",
    depends_on = {},
    produces = {"alignment_value"},
    status = "COMPLETE"
  },
  
  ["core/simulation.py"] = {
    function = "compute_entropy",
    purpose = "Compute entropy metric: Shannon entropy of vector distribution",
    depends_on = {},
    produces = {"entropy_value"},
    status = "COMPLETE"
  },
  
  ["core/physics.py"] = {
    function = "total_system_energy",
    purpose = "Compute total system energy using consistent Hamiltonian with Metropolis updates",
    depends_on = {},
    produces = {"energy_value"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 3 - Complete)
  ["tests/test_simulation.py"] = {
    function = "TestSimulation",
    purpose = "Test core simulation functionality (21 tests)",
    depends_on = {"core/simulation.py", "core/physics.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- 9. Dynamics and correlation analysis (Phase 4 - Complete)
  ["core/dynamics.py"] = {
    function = "compute_correlation_matrix",
    purpose = "Compute correlation matrix C_ij = cos(θ_ij) between normalized vectors",
    depends_on = {},
    produces = {"correlation_matrix"},
    status = "COMPLETE"
  },
  
  ["core/dynamics.py"] = {
    function = "compute_correlation_length",
    purpose = "Compute correlation length with optional linguistic distance weighting",
    depends_on = {"core/dynamics.py"},
    produces = {"correlation_length"},
    status = "COMPLETE"
  },
  
  ["core/dynamics.py"] = {
    function = "alignment_curvature",
    purpose = "Compute second derivative for Tc detection",
    depends_on = {},
    produces = {"curvature_array"},
    status = "COMPLETE"
  },
  
  -- 10. Phase detection and clustering (Phase 4 - Complete)
  ["core/phase_detection.py"] = {
    function = "find_critical_temperature",
    purpose = "Find critical temperature using Binder cumulant method (canonical)",
    depends_on = {"core/dynamics.py", "core/clustering.py"},
    produces = {"critical_temperature"},
    status = "COMPLETE"
  },
  
  ["core/phase_detection.py"] = {
    function = "detect_powerlaw_regime",
    purpose = "Detect power law in cluster size distribution",
    depends_on = {"core/clustering.py"},
    produces = {"power_law_metrics"},
    status = "COMPLETE"
  },
  
  ["core/clustering.py"] = {
    function = "cluster_vectors",
    purpose = "Cluster vectors based on cosine similarity threshold",
    depends_on = {},
    produces = {"cluster_list"},
    status = "COMPLETE"
  },
  
  -- 11. Meta vector computation (Phase 5 - Complete)
  ["core/meta_vector.py"] = {
    function = "compute_meta_vector",
    purpose = "Compute meta vector using specified method (comprehensive version with all methods)",
    depends_on = {},
    produces = {"meta_vector_result"},
    status = "COMPLETE"
  },
  
  ["core/meta_vector.py"] = {
    function = "compute_centroid",
    purpose = "Compute centroid (mean) of normalized vectors",
    depends_on = {},
    produces = {"centroid_vector"},
    status = "COMPLETE"
  },
  
  ["core/meta_vector.py"] = {
    function = "compute_medoid",
    purpose = "Find medoid (closest vector to centroid)",
    depends_on = {},
    produces = {"medoid_vector"},
    status = "COMPLETE"
  },
  
  ["core/meta_vector.py"] = {
    function = "compute_weighted_mean",
    purpose = "Compute weighted mean of normalized vectors",
    depends_on = {},
    produces = {"weighted_vector"},
    status = "COMPLETE"
  },
  
  ["core/meta_vector.py"] = {
    function = "compute_geometric_median",
    purpose = "Compute geometric median using Weiszfeld's algorithm",
    depends_on = {},
    produces = {"geometric_median_vector"},
    status = "COMPLETE"
  },
  
  ["core/meta_vector.py"] = {
    function = "compute_first_principal_component",
    purpose = "Compute first principal component as meta vector",
    depends_on = {},
    produces = {"pca_vector"},
    status = "COMPLETE"
  },
  
  -- 12. Comparison metrics (Phase 4.7 - Complete)
  ["core/comparison_metrics.py"] = {
    function = "compute_procrustes_distance",
    purpose = "Compute Procrustes distance between vector sets",
    depends_on = {},
    produces = {"procrustes_distance"},
    status = "COMPLETE"
  },
  
  ["core/comparison_metrics.py"] = {
    function = "compute_cka_similarity",
    purpose = "Compute Centered Kernel Alignment similarity",
    depends_on = {},
    produces = {"cka_similarity"},
    status = "COMPLETE"
  },
  
  ["core/comparison_metrics.py"] = {
    function = "compute_emd_distance",
    purpose = "Compute Earth Mover's Distance between distributions",
    depends_on = {},
    produces = {"emd_distance"},
    status = "COMPLETE"
  },
  
  ["core/comparison_metrics.py"] = {
    function = "compute_kl_divergence",
    purpose = "Compute KL divergence between vector distributions",
    depends_on = {},
    produces = {"kl_divergence"},
    status = "COMPLETE"
  },
  
  ["core/comparison_metrics.py"] = {
    function = "compare_anchor_to_multilingual",
    purpose = "Comprehensive anchor comparison at critical temperature",
    depends_on = {"core/comparison_metrics.py"},
    produces = {"comparison_metrics"},
    status = "COMPLETE"
  },
  
  -- 13. Test files (Phase 4 - Complete)
  ["tests/test_dynamics.py"] = {
    function = "test_correlation_length",
    purpose = "Test correlation length calculation",
    depends_on = {"core/dynamics.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_phase_detection.py"] = {
    function = "test_find_critical_temperature",
    purpose = "Test Tc detection with synthetic data",
    depends_on = {"core/phase_detection.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_meta_vector.py"] = {
    function = "test_meta_vector_methods",
    purpose = "Test all meta vector methods",
    depends_on = {"core/meta_vector.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 6 - Complete)
  ["tests/test_integration.py"] = {
    function = "TestCompletePipeline, TestValidationAgainstKnownResults, TestEdgeCases, TestPerformanceBenchmarks, TestMemoryUsage",
    purpose = "Integration tests for complete pipeline (19 tests)",
    depends_on = {"core/simulation.py", "core/phase_detection.py", "core/meta_vector.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_performance.py"] = {
    function = "TestPerformanceBenchmarks, TestScalability, TestMemoryUsage",
    purpose = "Performance benchmarks and scalability tests (30 tests)",
    depends_on = {"core/simulation.py", "core/phase_detection.py", "core/meta_vector.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/TEST_SUITE_STATUS.md"] = {
    function = "Test suite documentation",
    purpose = "Comprehensive documentation of test suite status and coverage",
    depends_on = {},
    produces = {"documentation"},
    status = "COMPLETE"
  },
  
  -- 14. Post-simulation analysis (Phase 7 - Complete)
  ["core/post_analysis.py"] = {
    function = "analyze_simulation_results",
    purpose = "Perform post-simulation analysis including anchor comparison",
    depends_on = {"core/comparison_metrics.py", "core/phase_detection.py"},
    produces = {"analysis_results"},
    status = "COMPLETE"
  },
  
  ["core/post_analysis.py"] = {
    function = "generate_visualization_data",
    purpose = "Prepare data for UI visualizations",
    depends_on = {},
    produces = {"visualization_data"},
    status = "COMPLETE"
  },
  
  ["core/post_analysis.py"] = {
    function = "interpret_analysis_results",
    purpose = "Generate human-readable interpretations of analysis results",
    depends_on = {},
    produces = {"interpretation_strings"},
    status = "COMPLETE"
  },
  
  ["core/post_analysis.py"] = {
    function = "validate_analysis_inputs",
    purpose = "Validate inputs for post-simulation analysis",
    depends_on = {},
    produces = {"validation_result"},
    status = "COMPLETE"
  },
  
  -- Test files (Phase 7 - Complete)
  ["tests/test_post_analysis.py"] = {
    function = "TestPostAnalysis",
    purpose = "Test post-analysis functionality (24 tests)",
    depends_on = {"core/post_analysis.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  -- 15. Export and I/O (Phase 8 - Complete)
  ["export/io.py"] = {
    function = "save_json",
    purpose = "Store results with proper error handling",
    depends_on = {},
    produces = {"json_file"},
    status = "COMPLETE"
  },
  
  ["export/io.py"] = {
    function = "save_csv",
    purpose = "Store results with proper error handling",
    depends_on = {},
    produces = {"csv_file"},
    status = "COMPLETE"
  },
  
  ["export/io.py"] = {
    function = "save_embeddings",
    purpose = "Save embeddings in multiple formats",
    depends_on = {"export/io.py"},
    produces = {"embedding_files"},
    status = "COMPLETE"
  },
  
  ["export/results.py"] = {
    function = "export_results",
    purpose = "Export all simulation results with anchor comparison",
    depends_on = {"export/io.py"},
    produces = {"export_files"},
    status = "COMPLETE"
  },
  
  ["export/ui_helpers.py"] = {
    function = "export_csv_results",
    purpose = "Export simulation results as CSV",
    depends_on = {},
    produces = {"csv_file"},
    status = "COMPLETE"
  },
  
  ["export/ui_helpers.py"] = {
    function = "export_vectors_at_tc",
    purpose = "Export vectors at critical temperature",
    depends_on = {},
    produces = {"npy_file"},
    status = "COMPLETE"
  },
  
  ["export/ui_helpers.py"] = {
    function = "export_charts",
    purpose = "Export charts as PNG files",
    depends_on = {},
    produces = {"png_file"},
    status = "COMPLETE"
  },
  
  ["export/ui_helpers.py"] = {
    function = "export_config_file",
    purpose = "Export current configuration as YAML",
    depends_on = {},
    produces = {"yaml_file"},
    status = "COMPLETE"
  },
  
  -- 16. UI components (Phase 9 - COMPLETE)
  ["ui/charts.py"] = {
    function = "plot_entropy_vs_temperature",
    purpose = "Plot entropy vs temperature with Tc marker",
    depends_on = {},
    produces = {"plotly_figure"},
    status = "COMPLETE"
  },
  
  ["ui/charts.py"] = {
    function = "plot_full_umap_projection",
    purpose = "Plot full UMAP projection of vectors at Tc with anchor highlighting",
    depends_on = {},
    produces = {"plotly_figure"},
    status = "COMPLETE"
  },
  
  ["ui/charts.py"] = {
    function = "plot_correlation_decay",
    purpose = "Plot correlation decay vs distance (log-log)",
    depends_on = {},
    produces = {"plotly_figure"},
    status = "COMPLETE"
  },
  
  ["ui/charts.py"] = {
    function = "plot_correlation_length_vs_temperature",
    purpose = "Plot correlation length vs temperature with Tc marker",
    depends_on = {},
    produces = {"plotly_figure"},
    status = "COMPLETE"
  },
  
  ["ui/components.py"] = {
    function = "render_anchor_config",
    purpose = "Anchor language configuration sidebar",
    depends_on = {},
    produces = {"anchor_language", "include_anchor"},
    status = "COMPLETE"
  },
  
  ["ui/components.py"] = {
    function = "render_experiment_description",
    purpose = "Display experiment configuration details",
    depends_on = {},
    produces = {"description_display"},
    status = "COMPLETE"
  },
  
  ["ui/tabs/simulation.py"] = {
    function = "render_simulation_tab",
    purpose = "Updated simulation tab with anchor configuration and metrics export",
    depends_on = {"ui/components.py", "main.py"},
    produces = {"simulation_display"},
    status = "COMPLETE"
  },
  
  ["ui/tabs/anchor_comparison.py"] = {
    function = "render_anchor_comparison_tab",
    purpose = "New tab for anchor comparison results with UMAP visualization",
    depends_on = {"ui/charts.py"},
    produces = {"comparison_display"},
    status = "COMPLETE"
  },
  
  ["ui/tabs/tabs.md"] = {
    function = "UI tabs documentation",
    purpose = "Document UI tab structure and functionality",
    depends_on = {},
    produces = {"documentation"},
    status = "COMPLETE"
  },
  
  -- Main Streamlit application
  ["app.py"] = {
    function = "main_streamlit_app",
    purpose = "Main Streamlit application with Overview, Simulation, and Anchor Comparison tabs",
    depends_on = {"ui/tabs/simulation.py", "ui/tabs/anchor_comparison.py", "ui/components.py"},
    produces = {"streamlit_interface"},
    status = "COMPLETE"
  },
  
  -- Test files for UI (Phase 9 - Complete)
  ["tests/test_ui.py"] = {
    function = "TestUI",
    purpose = "Test UI functionality and components",
    depends_on = {"ui/components.py", "ui/charts.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  },
  
  ["tests/test_ui_integration.py"] = {
    function = "TestUIIntegration",
    purpose = "Test UI integration with core simulation",
    depends_on = {"app.py", "core/simulation.py"},
    produces = {"test_results"},
    status = "COMPLETE"
  }
}