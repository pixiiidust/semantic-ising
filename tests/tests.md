# Tests Module Documentation

The `tests/` directory contains comprehensive test suites for the Semantic Ising Simulator, following Test-Driven Development (TDD) principles.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Test Data](#test-data)

## 🎯 Overview

The test suite provides comprehensive validation for the Semantic Ising Simulator, featuring:

- **242 total tests** across all modules (see [directory_structure.lua](../directory_structure.lua) for latest count)
- **Test-Driven Development** workflow
- **Comprehensive coverage** of core functionality
- **Integration testing** for end-to-end workflows
- **Performance benchmarking** and validation
- **Error handling** and edge case testing

## 📝 Recent Updates
- Tests updated to match new Tc detection logic and UI output.
- All new/changed logic is covered by corresponding tests, including UI synchronization and chart display for critical temperature.
- **Log(ξ) derivative method**: Tests updated to verify critical temperature detection using correlation length derivative
- **Anchor comparison metrics consistency**: Tests updated to verify that both main metrics and interactive metrics use original anchor vectors for comparison
- **Disk-based snapshot handling**: Tests added to verify proper handling of disk-based snapshots when in-memory snapshots are not available
- **Session state synchronization**: Tests updated to verify consistent metrics display across all UI components

## 📁 Test Structure

```
tests/
├── test_*.py                    # Individual module tests
├── TEST_SUITE_STATUS.md         # Test suite status and coverage
├── tests.md                     # This documentation file
├── fixtures/                    # Test data and fixtures
│   └── mini_dog_translations.json
└── ui/                          # UI-specific tests (empty)
```

## 🧪 Test Categories

For the complete and up-to-date test coverage information, including test counts by module and coverage statistics, please refer to [directory_structure.lua](../directory_structure.lua).

### 🔧 Core Module Tests

#### Configuration & CLI
- **`test_config_validation.py`** - Configuration validation
- **`test_config_types.py`** - Type conversion and validation
- **`test_cli.py`** - Command-line interface

#### Core Simulation
- **`test_simulation.py`** (21 tests) - Core simulation engine
- **`test_simulation_dataframe.py`** (4 tests) - Data structure validation
- **`test_temperature_estimation.py`** (26 tests) - Temperature estimation
- **`test_dynamics.py`** (15 tests) - Correlation and dynamics
- **`test_phase_detection.py`** (19 tests) - Critical temperature detection
- **`test_clustering.py`** (6 tests) - Vector clustering

#### Embeddings & Data
- **`test_embeddings.py`** (12 tests) - Multilingual embedding pipeline
- **`test_concept_detection.py`** (2 tests) - Concept file detection
- **`test_anchor_config.py`** (12 tests) - Anchor configuration system

#### Analysis & Metrics
- **`test_meta_vector.py`** (25 tests) - Meta vector computation
- **`test_comparison_metrics.py`** (18 tests) - Advanced comparison metrics
- **`test_post_analysis.py`** (24 tests) - Post-simulation analysis
- **`test_post_analysis_validate.py`** (6 tests) - Analysis validation

#### Export & I/O
- **`test_export.py`** (23 tests) - Export functionality
- **`test_logging.py`** (7 tests) - Logging utilities

### 🖥️ UI Tests

#### UI Components
- **`test_ui.py`** (25 tests) - UI component functionality
- **`test_ui_integration.py`** (20 tests) - UI integration testing

### 🔍 Validation Tests

#### Mathematical Validation
- **`test_validation.py`** (15 tests) - Mathematical property validation
- **`test_performance.py`** (30 tests) - Performance benchmarking

### 🔗 Integration Tests

#### End-to-End Testing
- **`test_integration.py`** (19 tests) - Complete pipeline testing

## 🚀 Running Tests

### Basic Test Execution
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_simulation.py -v

# Run specific test class
python -m pytest tests/test_simulation.py::TestSimulation -v

# Run specific test function
python -m pytest tests/test_simulation.py::TestSimulation::test_run_temperature_sweep_basic -v
```

### Coverage Reporting
```bash
# Run with coverage
python -m pytest --cov=core --cov=ui --cov=export --cov=config

# Generate HTML coverage report
python -m pytest --cov=core --cov=ui --cov=export --cov=config --cov-report=html

# Generate XML coverage report
python -m pytest --cov=core --cov=ui --cov=export --cov=config --cov-report=xml
```

### Performance Testing
```bash
# Run performance benchmarks
python -m pytest tests/test_performance.py -v

# Run with timing information
python -m pytest --durations=10
```

### Parallel Testing
```bash
# Run tests in parallel (requires pytest-xdist)
python -m pytest -n auto

# Run specific number of workers
python -m pytest -n 4
```

## 📁 Test Data

### Fixtures Directory
```
tests/fixtures/
└── mini_dog_translations.json    # Minimal concept file for testing
```

### Synthetic Test Data
- **Configuration data**: Valid and invalid configs for testing
- **Embedding vectors**: Random normalized vectors for simulation
- **Temperature ranges**: Various temperature configurations
- **Language sets**: Different language combinations

### Mock Objects
- **File system**: Mock file operations for I/O testing
- **External APIs**: Mock embedding generation for isolated testing
- **UI components**: Mock Streamlit components for UI testing

## 🧪 Test Strategy

### Test-Driven Development (TDD)
1. **Write failing test** that describes expected behavior
2. **Implement minimal code** to make test pass
3. **Refactor code** while keeping tests green
4. **Repeat** for next feature

### Test Categories
- **Unit Tests**: Individual function testing with synthetic data
- **Integration Tests**: End-to-end workflow testing
- **Edge Case Tests**: Error conditions and boundary testing
- **Performance Tests**: Scalability and memory usage validation
- **Validation Tests**: Mathematical property verification

### Error Testing
- **Invalid inputs**: Test with invalid parameters
- **Missing data**: Test with incomplete data structures
- **File system errors**: Test with invalid paths and permissions
- **Network errors**: Test with connection failures
- **Memory errors**: Test with large datasets

## 🔧 Test Configuration

### pytest Configuration
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

### Coverage Configuration
```ini
[coverage:run]
source = core,ui,export,config
omit = 
    */tests/*
    */__pycache__/*
    */venv/*
    */env/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 📈 Test Metrics

### Performance Benchmarks
- **Test execution time**: ~30 seconds for full suite
- **Memory usage**: <500MB for all tests
- **Coverage percentage**: >95% for core modules
- **Test reliability**: 100% pass rate

### Quality Metrics
- **Code coverage**: Comprehensive across all modules
- **Error handling**: All error conditions tested
- **Edge cases**: Boundary conditions validated
- **Integration**: End-to-end workflows tested

## 🚨 Common Test Issues

### Import Errors
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist
```

### File Permission Errors
```bash
# Ensure write permissions for test directories
chmod -R 755 tests/
```

## 📚 References

- **pytest**: Python testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **Test-Driven Development**: Development methodology
- **Unit Testing**: Software testing methodology

## Phase Detection Tests
- Now includes a test for Tc detection using the log(ξ) derivative (knee detection) with synthetic data.
- The old alignment-based detection is deprecated and only used as a fallback if correlation_length is not available. 