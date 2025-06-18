# Configuration Module Documentation

The `config/` directory contains configuration management utilities for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Files](#files)
- [Configuration Parameters](#configuration-parameters)
- [Validation](#validation)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The configuration module provides a robust, validated configuration system for the Semantic Ising Simulator, featuring:

- **YAML-based configuration** with comprehensive parameter coverage
- **Automatic validation** with detailed error messages
- **Type checking** and parameter bounds validation
- **Default values** for all parameters
- **API stability** (locked at v0.2)

## 📁 Files

### ⚙️ `defaults.yaml`
**Purpose**: Default configuration file with all simulation parameters

**Features**:
- Complete parameter coverage for all simulation aspects
- Anchor configuration for experimental design
- Temperature estimation and simulation parameters
- UMAP and clustering configuration
- Export and I/O settings

**API Status**: Locked at v0.2 (stable)

### 🔍 `validator.py`
**Purpose**: Configuration validation and loading utilities

**Key Functions**:
- `validate_config(config)` - Validates configuration dictionary
- `load_config(filepath)` - Loads and validates YAML configuration file
- `_ensure_float(value, param_name)` - Type conversion with error handling

**Features**:
- Comprehensive error handling and type checking
- Detailed error messages for invalid parameters
- Automatic type conversion for string parameters
- Parameter bounds validation

### 📄 `config.md`
**Purpose**: This documentation file describing the configuration module

## ⚙️ Configuration Parameters

### Temperature Settings
```yaml
temperature_range: [0.1, 3.0]      # Simulation temperature range
temperature_steps: 50               # Number of temperature points
```

### Encoder Configuration
```yaml
default_encoder: "sentence-transformers/LaBSE"  # Primary encoder model
supported_languages: ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
```

### Anchor Configuration
```yaml
anchor_config:
  default_anchor_language: "en"     # Default anchor language
  include_anchor_default: false     # Default anchor inclusion mode
```

### Simulation Parameters
```yaml
simulation_params:
  max_iterations: 1000              # Maximum simulation iterations
  convergence_threshold: 1e-6       # Convergence tolerance
  energy_coupling: 1.0              # Ising coupling strength
  update_method: "metropolis"       # Update rule (metropolis/glauber)
```

### Vector Storage
```yaml
store_all_temperatures: false       # Store vectors at all temperatures
```

### UMAP Parameters
```yaml
umap_params:
  n_neighbors: 15                   # UMAP neighborhood size
  min_dist: 0.1                     # Minimum distance
  n_components: 2                   # Output dimensions
  random_state: 42                  # Reproducibility seed
```

### Clustering Parameters
```yaml
cluster_params:
  similarity_threshold: 0.8         # Clustering threshold
  min_cluster_size: 2               # Minimum cluster size
```

### Correlation Analysis
```yaml
lambda_distance: 0.5                # Linguistic distance weighting
```

## 🔍 Validation

### Parameter Validation Rules
- **Temperature range**: Must be ascending, positive values
- **Temperature steps**: Must be ≥ 2
- **Anchor language**: Must be in supported languages list
- **Update method**: Must be "metropolis" or "glauber"
- **Thresholds**: Must be in valid ranges (0-1 for similarities)

### Type Validation
- **Automatic conversion**: String numbers converted to floats
- **Type checking**: Ensures correct data types
- **Bounds checking**: Validates parameter ranges
- **Required fields**: Ensures all necessary parameters present

### Error Handling
```python
# Invalid temperature range
ValueError: "Temperature range must be ascending"

# Missing required field
ValueError: "Missing required config key: temperature_range"

# Invalid anchor language
ValueError: "Default anchor language 'invalid' not in supported languages"
```

## 💡 Usage Examples

### Loading Default Configuration
```python
from config.validator import load_config

# Load default configuration
config = load_config("config/defaults.yaml")

# Access parameters
temp_range = config['temperature_range']
encoder = config['default_encoder']
anchor_lang = config['anchor_config']['default_anchor_language']
```

### Custom Configuration
```python
from config.validator import validate_config

# Custom configuration
custom_config = {
    'temperature_range': [0.5, 2.5],
    'temperature_steps': 30,
    'default_encoder': 'LaBSE',
    'anchor_config': {
        'default_anchor_language': 'es',
        'include_anchor_default': True
    }
}

# Validate custom configuration
validated_config = validate_config(custom_config)
```

### Configuration in Simulation
```python
from config.validator import load_config
from core.simulation import run_temperature_sweep

# Load configuration
config = load_config("config/defaults.yaml")

# Use in simulation
results = run_temperature_sweep(
    vectors=embeddings,
    T_range=config['temperature_range'],
    store_all_temperatures=config['store_all_temperatures']
)
```

### CLI Configuration
```python
import argparse
from config.validator import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/defaults.yaml')
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    return config
```

## 🔧 Configuration Management

### Environment-Specific Configs
```yaml
# development.yaml
simulation_params:
  max_iterations: 100  # Faster for development

# production.yaml  
simulation_params:
  max_iterations: 1000  # Full simulation
```

### Parameter Overrides
```python
# Override specific parameters
config = load_config("config/defaults.yaml")
config['temperature_range'] = [0.1, 5.0]  # Custom range
config['simulation_params']['max_iterations'] = 500  # Custom iterations
```

### Validation Workflow
1. **Load configuration** from YAML file
2. **Validate structure** and required fields
3. **Type conversion** for string parameters
4. **Bounds checking** for parameter values
5. **Return validated** configuration dictionary

## 🧪 Testing

Configuration validation has comprehensive test coverage:

- **Unit tests**: Individual validation functions
- **Integration tests**: End-to-end configuration loading
- **Error tests**: Invalid configuration handling
- **Type tests**: String-to-float conversion

Run configuration tests:
```bash
pytest tests/test_config_validation.py -v
pytest tests/test_config_types.py -v
```

## 📚 References

- **YAML**: Human-readable data serialization
- **PyYAML**: Python YAML parser and emitter
- **Type Hints**: Python type annotation system
- **Configuration Management**: Best practices for application configuration 