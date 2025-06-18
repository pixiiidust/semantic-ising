"""
Configuration validation and loading utilities for Semantic Ising Simulator.

This module provides functions to validate configuration parameters and load
configuration from YAML files with proper error handling.
"""

import yaml
from typing import Dict, Any, List


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters and provide defaults.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If required keys are missing or values are invalid
    """
    # Check required keys
    required_keys = ['temperature_range', 'temperature_steps', 'default_encoder']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate temperature range
    temp_range = config['temperature_range']
    if temp_range[0] >= temp_range[1]:
        raise ValueError("Temperature range must be ascending")
    
    # Validate temperature steps
    if config['temperature_steps'] < 2:
        raise ValueError("Temperature steps must be >= 2")
    
    # Validate anchor configuration if present
    if 'anchor_config' in config:
        anchor_config = config['anchor_config']
        
        if 'default_anchor_language' in anchor_config:
            default_lang = anchor_config['default_anchor_language']
            supported_langs = config.get('supported_languages', [])
            if supported_langs and default_lang not in supported_langs:
                raise ValueError(f"Default anchor language '{default_lang}' not in supported languages")
        
        if 'include_anchor_default' in anchor_config:
            if not isinstance(anchor_config['include_anchor_default'], bool):
                raise ValueError("include_anchor_default must be boolean")
    
    # Validate vector storage options
    if 'store_all_temperatures' in config:
        if not isinstance(config['store_all_temperatures'], bool):
            raise ValueError("store_all_temperatures must be boolean")
    
    return config


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        filepath: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or config validation fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Empty configuration file")
        
        return validate_config(config)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Error loading configuration: {e}") 