"""
Semantic Ising Simulator - Main CLI Interface

This module provides the command-line interface for running semantic Ising
simulations with configurable parameters and anchor language support.
"""

import argparse
import sys
import json
import numpy as np
from typing import List, Dict, Any

# Import functions we'll implement later
# from core.embeddings import generate_embeddings
# from core.simulation import run_temperature_sweep
# from core.phase_detection import find_critical_temperature
# from export.results import export_results


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
        
    Raises:
        SystemExit: If required arguments are missing or help is requested
    """
    parser = argparse.ArgumentParser(
        description="Semantic Ising Simulator - Multilingual embedding alignment under Ising dynamics"
    )
    
    # Required arguments
    parser.add_argument(
        "--concept", 
        required=True, 
        help="Concept name (e.g., 'dog')"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--encoder", 
        default="LaBSE", 
        help="Encoder model (default: LaBSE)"
    )
    parser.add_argument(
        "--t-min", 
        type=float, 
        default=0.1, 
        help="Minimum temperature (default: 0.1)"
    )
    parser.add_argument(
        "--t-max", 
        type=float, 
        default=3.0, 
        help="Maximum temperature (default: 3.0)"
    )
    parser.add_argument(
        "--t-steps", 
        type=int, 
        default=50, 
        help="Number of temperature steps (default: 50)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./results", 
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--config", 
        default="config/defaults.yaml", 
        help="Config file path (default: config/defaults.yaml)"
    )
    
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run simulation from CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Simulation results dictionary
        
    Raises:
        SystemExit: If simulation fails
    """
    try:
        # Generate temperature range
        T_range = list(np.linspace(args.t_min, args.t_max, args.t_steps))
        
        # TODO: Implement actual simulation
        # For now, return mock results
        result = {
            'concept': args.concept,
            'encoder': args.encoder,
            'temperature_range': T_range,
            'output_dir': args.output_dir,
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_simulation_from_file(config_file: str) -> Dict[str, Any]:
    """
    Run simulation from JSON configuration file.
    
    Args:
        config_file: Path to JSON configuration file
        
    Returns:
        Simulation results dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If JSON is invalid
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract parameters
        concept = config.get('concept')
        encoder = config.get('encoder', 'LaBSE')
        temp_range = config.get('temperature_range', [0.1, 3.0])
        temp_steps = config.get('temperature_steps', 50)
        output_dir = config.get('output_dir', './results')
        
        if not concept:
            raise ValueError("Concept is required in config file")
        
        # Generate temperature range
        T_range = list(np.linspace(temp_range[0], temp_range[1], temp_steps))
        
        # TODO: Implement actual simulation
        # For now, return mock results
        result = {
            'concept': concept,
            'encoder': encoder,
            'temperature_range': T_range,
            'output_dir': output_dir,
            'status': 'success'
        }
        
        return result
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def main():
    """Main CLI entry point."""
    args = parse_args()
    result = run_cli(args)
    print(f"Simulation completed for concept: {result['concept']}")


if __name__ == "__main__":
    main() 