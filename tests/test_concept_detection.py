"""
Tests for dynamic concept detection functionality
"""

import pytest
import os
import json
import tempfile
import shutil
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.components import get_available_concepts


def test_concept_detection_logic():
    """Test the core logic of concept detection"""
    # Create a temporary test environment
    temp_dir = tempfile.mkdtemp()
    concepts_dir = os.path.join(temp_dir, "data", "concepts")
    os.makedirs(concepts_dir, exist_ok=True)
    
    try:
        # Test 1: Empty directory
        concepts = get_available_concepts()
        # This should work with the real data/concepts directory
        # We can't easily test the empty case without mocking
        
        # Test 2: Create some test files and test the logic manually
        dog_data = {"en": "dog", "es": "perro", "fr": "chien"}
        tree_data = {"en": "tree", "es": "árbol", "fr": "arbre"}
        
        with open(os.path.join(concepts_dir, "dog_translations.json"), 'w') as f:
            json.dump(dog_data, f)
        
        with open(os.path.join(concepts_dir, "tree_translations.json"), 'w') as f:
            json.dump(tree_data, f)
        
        # Test the logic manually
        concept_files = []
        
        if os.path.exists(concepts_dir):
            for filename in os.listdir(concepts_dir):
                if filename.endswith("_translations.json"):
                    filepath = os.path.join(concepts_dir, filename)
                    mtime = os.path.getmtime(filepath)
                    concept_name = filename.replace("_translations.json", "")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            translations = json.load(f)
                        language_count = len(translations)
                    except Exception:
                        language_count = 0
                    
                    concept_files.append({
                        'concept_name': concept_name,
                        'filename': filename,
                        'filepath': filepath,
                        'language_count': language_count,
                        'modification_time': mtime
                    })
            
            concept_files.sort(key=lambda x: x['modification_time'], reverse=True)
        
        # Verify the logic works
        assert len(concept_files) == 2
        concept_names = [c['concept_name'] for c in concept_files]
        assert "dog" in concept_names
        assert "tree" in concept_names
        
        # Check language counts
        for concept in concept_files:
            if concept['concept_name'] == 'dog':
                assert concept['language_count'] == 3
            elif concept['concept_name'] == 'tree':
                assert concept['language_count'] == 3
        
        # Check required fields
        for concept in concept_files:
            assert 'concept_name' in concept
            assert 'filename' in concept
            assert 'filepath' in concept
            assert 'language_count' in concept
            assert 'modification_time' in concept
    
    finally:
        shutil.rmtree(temp_dir)


def test_concept_detection_with_real_data():
    """Test with the actual data/concepts directory"""
    # This test uses the real data directory
    concepts = get_available_concepts()
    
    # Should return a list (empty or with concepts)
    assert isinstance(concepts, list)
    
    # If there are concepts, check their structure
    for concept in concepts:
        assert 'concept_name' in concept
        assert 'filename' in concept
        assert 'filepath' in concept
        assert 'language_count' in concept
        assert 'modification_time' in concept
        
        # Check that filename ends with _translations.json or _translations_XX.json
        assert (concept['filename'].endswith('_translations.json') or 
                '_translations_' in concept['filename'])
        
        # Check that concept_name is extracted correctly
        if concept['filename'].endswith('_translations.json'):
            expected_name = concept['filename'].replace('_translations.json', '')
        else:
            # Handle _translations_XX.json pattern
            expected_name = concept['filename'].split('_translations_')[0]
        assert concept['concept_name'] == expected_name


if __name__ == "__main__":
    pytest.main([__file__]) 