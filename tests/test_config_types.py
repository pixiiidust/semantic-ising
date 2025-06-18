"""
Tests for configuration type validation to catch string-vs-float bugs before runtime.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.validator import load_config
from core.simulation import _ensure_float


@pytest.mark.parametrize("key", ["max_iterations", "convergence_threshold", "noise_sigma", "steps_per_T"])
def test_numeric_config_values(key):
    """Test that numeric config values are properly castable to float."""
    config = load_config("config/defaults.yaml")
    sim_params = config.get('simulation_params', {})
    
    if key in sim_params:
        # This should not raise if value is properly numeric
        val = _ensure_float(sim_params[key], key)
        assert isinstance(val, float), f"Config value {key}={sim_params[key]} should be float, got {type(val)}"
    else:
        # If key is missing, test with default value
        default_values = {
            "max_iterations": 6000,
            "convergence_threshold": 3e-3,
            "noise_sigma": 0.04,
            "steps_per_T": 6000
        }
        val = _ensure_float(default_values[key], key)
        assert isinstance(val, float)


def test_config_string_values_are_converted():
    """Test that string values in config are properly converted to floats."""
    # Test with string values that should be converted
    test_values = {
        "max_iterations": "6000",
        "convergence_threshold": "3e-3", 
        "noise_sigma": "0.04",
        "steps_per_T": "6000"
    }
    
    for key, string_value in test_values.items():
        val = _ensure_float(string_value, key)
        assert isinstance(val, float), f"String {string_value} should convert to float"
        assert val > 0, f"Converted value should be positive"


def test_config_invalid_values_raise_error():
    """Test that invalid config values raise appropriate errors."""
    invalid_values = ["not_a_number", "abc", "", None]
    
    for invalid_val in invalid_values:
        with pytest.raises(ValueError, match="must be numeric"):
            _ensure_float(invalid_val, "test_param")


def test_config_default_fallback():
    """Test that invalid values with defaults fall back gracefully."""
    result = _ensure_float("invalid", "test_param", default=1.0)
    assert result == 1.0
    assert isinstance(result, float)


if __name__ == "__main__":
    pytest.main([__file__]) 