"""
Anchor Configuration System for Semantic Ising Simulator

This module provides functions to configure which languages participate in
Ising dynamics vs. comparison experiments, supporting both single-phase
and two-phase experimental designs.
"""

from typing import List, Tuple


def configure_anchor_experiment(
    all_languages: List[str], 
    anchor_language: str, 
    include_anchor: bool
) -> Tuple[List[str], List[str]]:
    """
    Configure which languages participate in Ising dynamics vs. comparison.
    
    Args:
        all_languages: List of all available languages
        anchor_language: The reference language for comparison
        include_anchor: Whether to include anchor in dynamics (True) or compare to dynamics result (False)
    
    Returns:
        Tuple of (dynamics_languages, comparison_languages)
    
    Raises:
        ValueError: If anchor_language not in all_languages
    """
    if anchor_language not in all_languages:
        raise ValueError(f"Anchor language '{anchor_language}' not found in available languages")
    
    if include_anchor:
        # Single-phase: anchor included in dynamics
        dynamics_languages = all_languages.copy()
        comparison_languages = [anchor_language]
    else:
        # Two-phase: anchor excluded from dynamics
        dynamics_languages = [lang for lang in all_languages if lang != anchor_language]
        comparison_languages = [anchor_language]
    
    return dynamics_languages, comparison_languages


def validate_anchor_config(
    all_languages: List[str], 
    anchor_language: str, 
    include_anchor: bool
) -> bool:
    """
    Validate anchor configuration parameters.
    
    Args:
        all_languages: List of all available languages
        anchor_language: The reference language for comparison
        include_anchor: Whether to include anchor in dynamics
    
    Returns:
        True if configuration is valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    if anchor_language not in all_languages:
        raise ValueError(f"Anchor language '{anchor_language}' not found in available languages")
    
    if not include_anchor and len(all_languages) <= 1:
        raise ValueError("Cannot exclude anchor when only one language available")
    
    return True


def get_experiment_description(
    anchor_language: str, 
    include_anchor: bool, 
    dynamics_languages: List[str]
) -> str:
    """
    Generate human-readable experiment description.
    
    Args:
        anchor_language: The reference language for comparison
        include_anchor: Whether anchor participates in dynamics
        dynamics_languages: Languages participating in Ising dynamics
    
    Returns:
        Human-readable description of experiment configuration
    """
    if include_anchor:
        return f"Single-phase experiment: {anchor_language} participates in Ising dynamics with {len(dynamics_languages)} languages"
    else:
        return f"Two-phase experiment: {anchor_language} compared to Ising dynamics of {len(dynamics_languages)} languages" 