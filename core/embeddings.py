"""
Core embedding functionality for the Semantic Ising Simulator.

This module handles loading multilingual concept translations, generating
embeddings using sentence transformers, and caching results for efficiency.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple

# Initialize logger
logger = logging.getLogger(__name__)


def load_concept_embeddings(concept_name: str, filename: str = None) -> Dict[str, str]:
    """
    Read multilingual terms with error handling.
    
    Args:
        concept_name: Name of the concept (e.g., 'dog')
        filename: Optional specific filename to load (e.g., 'dog_translations_72.json')
        
    Returns:
        Dictionary mapping language codes to translations
        
    Raises:
        FileNotFoundError: If translation file not found
        ValueError: If file is empty or contains invalid JSON
    """
    if filename:
        # Use the specific filename provided
        filepath = f"data/concepts/{filename}"
    else:
        # Fallback to the old pattern
        filepath = f"data/concepts/{concept_name}_translations.json"
        print(f"Using fallback filepath: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            translations = json.load(f)
            
        if not translations:
            raise ValueError(f"Empty translations file for concept: {concept_name}")
            
        return translations
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Translation file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in translation file: {filepath}")


def generate_embeddings(concept_name: str, encoder_name: str, filename: str = None) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings with caching and error recovery.
    
    Args:
        concept_name: Name of the concept
        encoder_name: Name of the sentence transformer model
        filename: Optional specific filename to load
        
    Returns:
        Tuple of (embeddings_array, language_list)
        
    Raises:
        RuntimeError: If embedding generation fails
    """
    # Create cache key that includes filename if provided
    cache_key = f"{concept_name}_{encoder_name}"
    if filename:
        # Use filename without extension for cache key
        cache_key += f"_{filename.replace('.json', '')}"
    
    cache_path = f"data/embeddings/{cache_key}_cached.npy"
    
    if os.path.exists(cache_path):
        try:
            embeddings = np.load(cache_path)
            translations = load_concept_embeddings(concept_name, filename)
            
            # Ensure vectors are normalized
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings, list(translations.keys())
            
        except Exception as e:
            logger.warning(f"Cache corrupted, regenerating: {e}")
    
    # Generate new embeddings
    translations = load_concept_embeddings(concept_name, filename)
    
    # Check if we should use mock embeddings due to PyTorch issues
    use_mock_embeddings = False
    
    try:
        # Try to import sentence transformers with comprehensive error handling
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as import_error:
            logger.warning(f"Sentence transformers import failed: {import_error}")
            use_mock_embeddings = True
        except Exception as e:
            if "torch" in str(e).lower() or "torch._classes" in str(e) or "meta tensor" in str(e).lower():
                logger.warning(f"PyTorch-related import error: {e}")
                use_mock_embeddings = True
            else:
                raise ImportError(f"Unexpected import error: {e}")
        
        if not use_mock_embeddings:
            # Try to create model with comprehensive error handling
            try:
                # Force CPU device to avoid meta tensor issues
                model = SentenceTransformer(encoder_name, device='cpu')
            except Exception as model_error:
                if any(keyword in str(model_error).lower() for keyword in ["meta tensor", "torch.nn.Module.to_empty", "custom class", "torch._classes"]):
                    logger.warning(f"PyTorch device/custom class error: {model_error}")
                    use_mock_embeddings = True
                else:
                    raise RuntimeError(f"Model creation failed: {model_error}")
            
            if not use_mock_embeddings:
                texts = list(translations.values())
                
                # Try to encode with comprehensive error handling
                try:
                    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, device='cpu')
                except Exception as encode_error:
                    if any(keyword in str(encode_error).lower() for keyword in ["meta tensor", "custom class", "torch._classes"]):
                        logger.warning(f"PyTorch encoding error: {encode_error}")
                        use_mock_embeddings = True
                    else:
                        raise RuntimeError(f"Encoding failed: {encode_error}")
                
                if not use_mock_embeddings:
                    # Normalize embeddings to unit length
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # Cache embeddings
                    cache_embeddings(embeddings, cache_key, encoder_name)
                    
                    return embeddings, list(translations.keys())
        
    except Exception as e:
        logger.warning(f"Unexpected error in embedding generation: {e}")
        use_mock_embeddings = True
    
    # Create mock embeddings for testing when PyTorch fails
    if use_mock_embeddings:
        raise RuntimeError(
            "Failed to generate real embeddings due to PyTorch/sentence-transformers issues. "
            "Check your environment, PyTorch, and sentence-transformers installation. "
            "Mock embeddings are disabled for production use."
        )
    
    # This should never be reached, but just in case
    raise RuntimeError("Failed to generate embeddings: Unknown error")


def cache_embeddings(embeddings: np.ndarray, concept: str, encoder: str) -> str:
    """
    Store embeddings with validation.
    
    Args:
        embeddings: 2D numpy array of embeddings
        concept: Concept name (or cache key)
        encoder: Encoder model name
        
    Returns:
        Path to the cached file
        
    Raises:
        ValueError: If embeddings are invalid
    """
    cache_dir = "data/embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"{concept}_{encoder}_cached.npy"
    filepath = os.path.join(cache_dir, filename)
    
    # Validate embeddings before caching
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array")
    
    if embeddings.shape[0] == 0:
        raise ValueError("Empty embeddings array")
    
    np.save(filepath, embeddings)
    return filepath 