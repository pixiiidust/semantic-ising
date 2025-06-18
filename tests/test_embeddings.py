import pytest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Import the functions we'll implement
from core.embeddings import load_concept_embeddings, generate_embeddings, cache_embeddings


class TestEmbeddings:
    """Test embedding functionality"""
    
    def test_load_concept_embeddings_valid(self):
        """Test that valid concept file loads correctly"""
        # Test with actual dog file that exists
        result = load_concept_embeddings("dog")
        expected_keys = ["en", "es", "fr", "de", "it", "pt"]
        
        # Check that we get the expected languages
        for key in expected_keys:
            assert key in result
        assert result["en"] == "dog"
        assert result["es"] == "perro"
    
    def test_load_concept_embeddings_missing_file(self):
        """Test that missing concept file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Translation file not found"):
            load_concept_embeddings("nonexistent_concept")
    
    def test_load_concept_embeddings_invalid_json(self):
        """Test that invalid JSON raises ValueError"""
        # Create the invalid JSON file in the correct location
        invalid_file = "data/concepts/invalid_concept_translations.json"
        os.makedirs(os.path.dirname(invalid_file), exist_ok=True)
        
        with open(invalid_file, 'w') as f:
            f.write("invalid: json: content: [")
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON in translation file"):
                load_concept_embeddings("invalid_concept")
        finally:
            if os.path.exists(invalid_file):
                os.unlink(invalid_file)
    
    def test_load_concept_embeddings_empty_file(self):
        """Test that empty translations file raises ValueError"""
        # Create the empty JSON file in the correct location
        empty_file = "data/concepts/empty_concept_translations.json"
        os.makedirs(os.path.dirname(empty_file), exist_ok=True)
        
        with open(empty_file, 'w') as f:
            json.dump({}, f)
        
        try:
            with pytest.raises(ValueError, match="Empty translations file for concept"):
                load_concept_embeddings("empty_concept")
        finally:
            if os.path.exists(empty_file):
                os.unlink(empty_file)
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('core.embeddings.load_concept_embeddings')
    @patch('core.embeddings.cache_embeddings')
    @patch('core.embeddings.os.path.exists')
    def test_generate_embeddings_success(self, mock_exists, mock_cache, mock_load, mock_transformer):
        """Test successful embedding generation"""
        # Mock cache does not exist
        mock_exists.return_value = False
        
        # Mock concept data
        concept_data = {
            "en": "dog",
            "es": "perro",
            "fr": "chien"
        }
        mock_load.return_value = concept_data
        
        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 768)
        mock_transformer.return_value = mock_model
        
        # Mock cache function
        mock_cache.return_value = os.path.join("data", "embeddings", "dog_LaBSE_cached.npy")
        
        # Test function
        embeddings, languages = generate_embeddings("dog", "LaBSE")
        
        # Verify results
        assert embeddings.shape == (3, 768)
        assert languages == ["en", "es", "fr"]
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)  # Normalized
        
        # Verify calls
        mock_load.assert_called_once_with("dog", None)
        mock_transformer.assert_called_once_with("LaBSE")
        mock_model.encode.assert_called_once_with(
            ["dog", "perro", "chien"], 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        mock_cache.assert_called_once()
    
    @patch('core.embeddings.os.path.exists')
    @patch('core.embeddings.np.load')
    @patch('core.embeddings.load_concept_embeddings')
    def test_generate_embeddings_caching(self, mock_load, mock_np_load, mock_exists):
        """Test that cached embeddings are loaded when available"""
        # Mock cache exists
        mock_exists.return_value = True
        
        # Mock cached embeddings
        cached_embeddings = np.random.randn(3, 768)
        cached_embeddings = cached_embeddings / np.linalg.norm(cached_embeddings, axis=1, keepdims=True)
        mock_np_load.return_value = cached_embeddings
        
        # Mock concept data
        concept_data = {
            "en": "dog",
            "es": "perro",
            "fr": "chien"
        }
        mock_load.return_value = concept_data
        
        # Test function
        embeddings, languages = generate_embeddings("dog", "LaBSE")
        
        # Verify results
        assert embeddings.shape == (3, 768)
        assert languages == ["en", "es", "fr"]
        assert np.allclose(embeddings, cached_embeddings)
        
        # Verify cache was checked
        mock_exists.assert_called_once()
        mock_np_load.assert_called_once()
    
    @patch('core.embeddings.os.path.exists')
    @patch('core.embeddings.np.load')
    @patch('core.embeddings.load_concept_embeddings')
    def test_generate_embeddings_corrupted_cache(self, mock_load, mock_np_load, mock_exists):
        """Test that corrupted cache is handled gracefully"""
        # Mock cache exists but is corrupted
        mock_exists.return_value = True
        mock_np_load.side_effect = Exception("Corrupted cache")
        
        # Mock concept data
        concept_data = {
            "en": "dog",
            "es": "perro"
        }
        mock_load.return_value = concept_data
        
        # Mock SentenceTransformer for regeneration
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(2, 768)
            mock_transformer.return_value = mock_model
            
            with patch('core.embeddings.cache_embeddings') as mock_cache:
                mock_cache.return_value = os.path.join("data", "embeddings", "dog_LaBSE_cached.npy")
                
                # Test function
                embeddings, languages = generate_embeddings("dog", "LaBSE")
                
                # Verify results
                assert embeddings.shape == (2, 768)
                assert languages == ["en", "es"]
    
    @patch('core.embeddings.load_concept_embeddings')
    @patch('sentence_transformers.SentenceTransformer')
    def test_generate_embeddings_encoder_error(self, mock_transformer, mock_load):
        """Test that encoder errors are handled properly"""
        # Mock concept data
        concept_data = {
            "en": "dog",
            "es": "perro"
        }
        mock_load.return_value = concept_data
        
        # Mock SentenceTransformer error
        mock_transformer.side_effect = Exception("Encoder failed")
        
        # Test function
        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            generate_embeddings("dog", "invalid_encoder")
    
    def test_cache_embeddings_validation(self):
        """Test that embeddings are validated before caching"""
        # Test invalid embeddings (1D array)
        invalid_embeddings = np.random.randn(768)
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            cache_embeddings(invalid_embeddings, "dog", "LaBSE")
        
        # Test empty embeddings (0D array)
        empty_embeddings = np.array([])
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            cache_embeddings(empty_embeddings, "dog", "LaBSE")
    
    @patch('core.embeddings.os.makedirs')
    @patch('core.embeddings.np.save')
    def test_cache_embeddings_success(self, mock_save, mock_makedirs):
        """Test successful embedding caching"""
        # Create valid embeddings
        embeddings = np.random.randn(3, 768)
        
        # Test function
        result = cache_embeddings(embeddings, "dog", "LaBSE")
        
        # Verify results (use os.path.normpath for cross-platform compatibility)
        expected_path = os.path.normpath(os.path.join("data", "embeddings", "dog_LaBSE_cached.npy"))
        assert os.path.normpath(result) == expected_path
        
        # Verify calls
        mock_makedirs.assert_called_once_with("data/embeddings", exist_ok=True)
        mock_save.assert_called_once()
    
    def test_cache_embeddings_filepath_generation(self):
        """Test that cache filepath is generated correctly"""
        embeddings = np.random.randn(2, 768)
        
        with patch('core.embeddings.os.makedirs'):
            with patch('core.embeddings.np.save'):
                result = cache_embeddings(embeddings, "cat", "BERT")
                expected_path = os.path.normpath(os.path.join("data", "embeddings", "cat_BERT_cached.npy"))
                assert os.path.normpath(result) == expected_path
    
    @patch('core.embeddings.load_concept_embeddings')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('core.embeddings.cache_embeddings')
    @patch('core.embeddings.os.path.exists')
    def test_generate_embeddings_normalization(self, mock_exists, mock_cache, mock_transformer, mock_load):
        """Test that embeddings are properly normalized"""
        # Mock cache does not exist
        mock_exists.return_value = False
        
        # Mock concept data
        concept_data = {
            "en": "dog",
            "es": "perro"
        }
        mock_load.return_value = concept_data
        
        # Mock SentenceTransformer with non-normalized embeddings
        mock_model = MagicMock()
        raw_embeddings = np.random.randn(2, 768) * 10  # Large values
        mock_model.encode.return_value = raw_embeddings
        mock_transformer.return_value = mock_model
        
        # Mock cache function
        mock_cache.return_value = os.path.join("data", "embeddings", "dog_LaBSE_cached.npy")
        
        # Test function
        embeddings, languages = generate_embeddings("dog", "LaBSE")
        
        # Verify normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # Verify embeddings are different from raw (due to normalization)
        # Use a more robust comparison that accounts for shape differences
        assert embeddings.shape != raw_embeddings.shape or not np.allclose(embeddings, raw_embeddings) 