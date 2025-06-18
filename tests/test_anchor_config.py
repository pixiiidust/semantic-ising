import pytest
from typing import List, Tuple

# Import the functions we'll implement
from core.anchor_config import (
    configure_anchor_experiment,
    validate_anchor_config,
    get_experiment_description
)


class TestAnchorConfig:
    """Test anchor configuration functionality"""
    
    def test_configure_anchor_experiment_single_phase(self):
        """Test single-phase configuration where anchor participates in dynamics"""
        all_languages = ["en", "es", "fr", "de", "it"]
        anchor_language = "en"
        include_anchor = True
        
        dynamics_languages, comparison_languages = configure_anchor_experiment(
            all_languages, anchor_language, include_anchor
        )
        
        # In single-phase mode, anchor should be in dynamics
        assert anchor_language in dynamics_languages
        assert len(dynamics_languages) == len(all_languages)
        assert set(dynamics_languages) == set(all_languages)
        assert comparison_languages == [anchor_language]
    
    def test_configure_anchor_experiment_two_phase(self):
        """Test two-phase configuration where anchor is compared to dynamics result"""
        all_languages = ["en", "es", "fr", "de", "it"]
        anchor_language = "en"
        include_anchor = False
        
        dynamics_languages, comparison_languages = configure_anchor_experiment(
            all_languages, anchor_language, include_anchor
        )
        
        # In two-phase mode, anchor should be excluded from dynamics
        assert anchor_language not in dynamics_languages
        assert len(dynamics_languages) == len(all_languages) - 1
        assert set(dynamics_languages) == set(["es", "fr", "de", "it"])
        assert comparison_languages == [anchor_language]
    
    def test_configure_anchor_experiment_different_anchor(self):
        """Test configuration with different anchor language"""
        all_languages = ["en", "es", "fr", "de", "it"]
        anchor_language = "es"
        include_anchor = False
        
        dynamics_languages, comparison_languages = configure_anchor_experiment(
            all_languages, anchor_language, include_anchor
        )
        
        assert anchor_language not in dynamics_languages
        assert set(dynamics_languages) == set(["en", "fr", "de", "it"])
        assert comparison_languages == [anchor_language]
    
    def test_validate_anchor_config_valid(self):
        """Test valid anchor configuration"""
        all_languages = ["en", "es", "fr", "de"]
        anchor_language = "en"
        include_anchor = True
        
        result = validate_anchor_config(all_languages, anchor_language, include_anchor)
        assert result is True
    
    def test_validate_anchor_config_invalid_language(self):
        """Test validation with anchor language not in available languages"""
        all_languages = ["en", "es", "fr", "de"]
        anchor_language = "zh"  # Not in available languages
        include_anchor = True
        
        with pytest.raises(ValueError, match="Anchor language 'zh' not found in available languages"):
            validate_anchor_config(all_languages, anchor_language, include_anchor)
    
    def test_validate_anchor_config_single_language_exclude(self):
        """Test validation when trying to exclude anchor with only one language"""
        all_languages = ["en"]
        anchor_language = "en"
        include_anchor = False  # Try to exclude when only one language
        
        with pytest.raises(ValueError, match="Cannot exclude anchor when only one language available"):
            validate_anchor_config(all_languages, anchor_language, include_anchor)
    
    def test_validate_anchor_config_single_language_include(self):
        """Test validation when including anchor with only one language (should pass)"""
        all_languages = ["en"]
        anchor_language = "en"
        include_anchor = True
        
        result = validate_anchor_config(all_languages, anchor_language, include_anchor)
        assert result is True
    
    def test_get_experiment_description_single_phase(self):
        """Test experiment description for single-phase mode"""
        anchor_language = "en"
        include_anchor = True
        dynamics_languages = ["en", "es", "fr", "de", "it"]
        
        description = get_experiment_description(anchor_language, include_anchor, dynamics_languages)
        
        expected = "Single-phase experiment: en participates in Ising dynamics with 5 languages"
        assert description == expected
    
    def test_get_experiment_description_two_phase(self):
        """Test experiment description for two-phase mode"""
        anchor_language = "en"
        include_anchor = False
        dynamics_languages = ["es", "fr", "de", "it"]
        
        description = get_experiment_description(anchor_language, include_anchor, dynamics_languages)
        
        expected = "Two-phase experiment: en compared to Ising dynamics of 4 languages"
        assert description == expected
    
    def test_get_experiment_description_single_language(self):
        """Test experiment description with single language in dynamics"""
        anchor_language = "en"
        include_anchor = True
        dynamics_languages = ["en"]
        
        description = get_experiment_description(anchor_language, include_anchor, dynamics_languages)
        
        expected = "Single-phase experiment: en participates in Ising dynamics with 1 languages"
        assert description == expected
    
    def test_configure_anchor_experiment_edge_case_empty_languages(self):
        """Test edge case with empty language list"""
        all_languages = []
        anchor_language = "en"
        include_anchor = True
        
        with pytest.raises(ValueError, match="Anchor language 'en' not found in available languages"):
            configure_anchor_experiment(all_languages, anchor_language, include_anchor)
    
    def test_configure_anchor_experiment_edge_case_duplicate_languages(self):
        """Test edge case with duplicate languages in list"""
        all_languages = ["en", "es", "en", "fr"]  # Duplicate "en"
        anchor_language = "en"
        include_anchor = False
        
        dynamics_languages, comparison_languages = configure_anchor_experiment(
            all_languages, anchor_language, include_anchor
        )
        
        # Should handle duplicates gracefully
        assert anchor_language not in dynamics_languages
        assert set(dynamics_languages) == set(["es", "fr"])
        assert comparison_languages == [anchor_language] 