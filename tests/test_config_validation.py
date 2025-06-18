import pytest
import tempfile
import os
import yaml
from unittest.mock import patch

# Import the functions we'll implement
from config.validator import validate_config, load_config


class TestConfigValidation:
    """Test config validation functionality"""
    
    def test_validate_config_valid_input(self):
        """Test that valid config passes validation"""
        valid_config = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE",
            'supported_languages': ["en", "es", "fr", "de"],
            'anchor_config': {
                'default_anchor_language': "en",
                'include_anchor_default': False
            },
            'store_all_temperatures': False
        }
        
        result = validate_config(valid_config)
        assert result == valid_config
    
    def test_validate_config_missing_required_keys(self):
        """Test that missing required keys raises ValueError"""
        invalid_config = {
            'temperature_range': [0.1, 3.0]
            # Missing temperature_steps and default_encoder
        }
        
        with pytest.raises(ValueError, match="Missing required config key"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_temperature_range(self):
        """Test that invalid temperature range raises ValueError"""
        invalid_config = {
            'temperature_range': [3.0, 0.1],  # Descending order
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE"
        }
        
        with pytest.raises(ValueError, match="Temperature range must be ascending"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_temperature_steps(self):
        """Test that invalid temperature steps raises ValueError"""
        invalid_config = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 1,  # Must be >= 2
            'default_encoder': "sentence-transformers/LaBSE"
        }
        
        with pytest.raises(ValueError, match="Temperature steps must be >= 2"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_anchor_language(self):
        """Test that invalid anchor language raises ValueError"""
        invalid_config = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE",
            'supported_languages': ["en", "es", "fr"],
            'anchor_config': {
                'default_anchor_language': "zh",  # Not in supported languages
                'include_anchor_default': False
            }
        }
        
        with pytest.raises(ValueError, match="Default anchor language 'zh' not in supported languages"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_include_anchor_type(self):
        """Test that invalid include_anchor_default type raises ValueError"""
        invalid_config = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE",
            'anchor_config': {
                'default_anchor_language': "en",
                'include_anchor_default': "not_a_boolean"  # Should be boolean
            }
        }
        
        with pytest.raises(ValueError, match="include_anchor_default must be boolean"):
            validate_config(invalid_config)
    
    def test_validate_config_invalid_store_all_temperatures_type(self):
        """Test that invalid store_all_temperatures type raises ValueError"""
        invalid_config = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE",
            'store_all_temperatures': "not_a_boolean"  # Should be boolean
        }
        
        with pytest.raises(ValueError, match="store_all_temperatures must be boolean"):
            validate_config(invalid_config)
    
    def test_load_config_from_file(self):
        """Test loading config from YAML file"""
        config_data = {
            'temperature_range': [0.1, 3.0],
            'temperature_steps': 50,
            'default_encoder': "sentence-transformers/LaBSE"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            assert result == config_data
        finally:
            os.unlink(temp_file)
    
    def test_load_config_file_not_found(self):
        """Test that missing config file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test that invalid YAML raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config(temp_file)
        finally:
            os.unlink(temp_file) 