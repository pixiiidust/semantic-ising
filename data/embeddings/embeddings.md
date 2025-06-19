# Embeddings Directory Documentation

The `data/embeddings/` directory contains cached embedding files for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Caching Strategy](#caching-strategy)
- [File Formats](#file-formats)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The embeddings directory serves as a cache for multilingual embeddings, providing:

- **Cached embeddings** to avoid expensive recomputation
- **Multiple encoder support** (LaBSE, SBERT, XLM-R)
- **Version control** for different language sets
- **Automatic validation** and error recovery

## 📁 File Structure

```
data/embeddings/
├── dog_LaBSE_cached.npy                                    # Legacy dog embeddings (LaBSE)
├── dog_LaBSE_dog_translations_LaBSE_cached.npy             # Legacy dog embeddings
├── dog_LaBSE_dog_translations_72_LaBSE_cached.npy          # Legacy extended dog embeddings (72 languages)
├── dog_25.json_LaBSE_dog_translations_25_LaBSE_cached.npy  # Dog embeddings (25 languages)
├── tree_LaBSE_tree_translations_LaBSE_cached.npy           # Legacy tree embeddings
├── tree_LaBSE_tree_translations_72_LaBSE_cached.npy        # Legacy extended tree embeddings (72 languages)
├── i_love_you_25.json_LaBSE_i_love_you_translations_25_LaBSE_cached.npy # I love you embeddings (25 languages)
├── i_love_you_75.json_LaBSE_i_love_you_translations_75_LaBSE_cached.npy # I love you embeddings (75 languages)
├── love_LaBSE_love_translations_75_LaBSE_cached.npy        # Love embeddings (75 languages)
└── embeddings.md                                           # This documentation file
```

## 🧠 Caching Strategy

### Automatic Caching
- **First-time generation**: Embeddings computed and cached automatically
- **Subsequent access**: Loaded from cache for instant access
- **Cache validation**: Corrupted files detected and regenerated
- **Version control**: Different caches for different language sets

### Cache File Properties
- **Format**: NumPy arrays (.npy)
- **Shape**: (n_languages, embedding_dimension)
- **Data type**: float32 for efficiency
- **Normalization**: Unit-length vectors
- **Size**: 75-225 KB per file

### File Naming Convention
- **Format**: `{concept}_{encoder}_{concept_file}_{encoder}_cached.npy`
- **Example**: `dog_LaBSE_dog_translations_72_LaBSE_cached.npy`

## 📄 File Formats

### NumPy Format (.npy)
```python
import numpy as np

# Load cached embeddings
embeddings = np.load("data/embeddings/dog_LaBSE_cached.npy")

# Properties
print(f"Shape: {embeddings.shape}")  # (n_languages, embedding_dim)
print(f"Data type: {embeddings.dtype}")  # float32
print(f"Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")  # True
```

### Supported Encoders
- **LaBSE**: Language-agnostic BERT Sentence Embedding (primary)
- **SBERT**: Sentence-BERT (planned)
- **XLM-R**: XLM-RoBERTa (planned)

### Available Cache Files
- `dog_LaBSE_cached.npy` - Legacy dog embeddings (LaBSE)
- `dog_LaBSE_dog_translations_LaBSE_cached.npy` - Legacy dog embeddings
- `dog_LaBSE_dog_translations_72_LaBSE_cached.npy` - Legacy extended dog embeddings (72 languages)
- `dog_25.json_LaBSE_dog_translations_25_LaBSE_cached.npy` - Dog embeddings (25 languages)
- `tree_LaBSE_tree_translations_LaBSE_cached.npy` - Legacy tree embeddings
- `tree_LaBSE_tree_translations_72_LaBSE_cached.npy` - Legacy extended tree embeddings (72 languages)
- `i_love_you_25.json_LaBSE_i_love_you_translations_25_LaBSE_cached.npy` - I love you embeddings (25 languages)
- `i_love_you_75.json_LaBSE_i_love_you_translations_75_LaBSE_cached.npy` - I love you embeddings (75 languages)
- `love_LaBSE_love_translations_75_LaBSE_cached.npy` - Love embeddings (75 languages)

## 💡 Usage Examples

### Loading Cached Embeddings
```python
import numpy as np
import os

def load_cached_embeddings(concept_name, encoder="LaBSE"):
    """Load cached embeddings for a concept."""
    cache_path = f"data/embeddings/{concept_name}_{encoder}_cached.npy"
    
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        return embeddings
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

# Usage
embeddings = load_cached_embeddings("dog", "LaBSE")
print(f"Embedding shape: {embeddings.shape}")
print(f"Number of languages: {embeddings.shape[0]}")
```

### Automatic Embedding Generation
```python
from core.embeddings import generate_embeddings

# Generate embeddings (will use cache if available)
embeddings, languages = generate_embeddings("dog", "LaBSE")

print(f"Generated embeddings for {len(languages)} languages")
print(f"Languages: {languages}")
```

### Cache Management
```python
import os
import numpy as np

def list_cache_files():
    """List all available cache files."""
    cache_dir = "data/embeddings"
    cache_files = []
    
    for filename in os.listdir(cache_dir):
        if filename.endswith("_cached.npy"):
            cache_files.append(filename)
    
    return sorted(cache_files)

def get_cache_info(filename):
    """Get information about a cache file."""
    filepath = f"data/embeddings/{filename}"
    
    if os.path.exists(filepath):
        embeddings = np.load(filepath)
        return {
            'filename': filename,
            'shape': embeddings.shape,
            'size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'languages': embeddings.shape[0],
            'dimensions': embeddings.shape[1]
        }
    else:
        return None

# Usage
cache_files = list_cache_files()
for filename in cache_files:
    info = get_cache_info(filename)
    if info:
        print(f"{info['filename']}: {info['languages']} languages, {info['dimensions']}D, {info['size_mb']:.2f}MB")
```

### Cache Validation
```python
def validate_cache_file(filepath):
    """Validate a cache file."""
    try:
        embeddings = np.load(filepath)
        
        # Check shape
        if embeddings.ndim != 2:
            return False, "Embeddings must be 2D array"
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            return False, "Vectors not normalized"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            return False, "Contains NaN or infinite values"
        
        return True, f"Valid cache with {embeddings.shape[0]} languages"
        
    except Exception as e:
        return False, f"Error loading cache: {e}"

# Usage
is_valid, message = validate_cache_file("data/embeddings/dog_LaBSE_cached.npy")
print(f"Validation: {message}")
```

### Cache Cleanup
```python
import os
import glob

def cleanup_old_cache_files(days_old=30):
    """Remove cache files older than specified days."""
    import time
    
    cache_dir = "data/embeddings"
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_files = []
    for filepath in glob.glob(f"{cache_dir}/*_cached.npy"):
        if os.path.getmtime(filepath) < cutoff_time:
            os.remove(filepath)
            removed_files.append(os.path.basename(filepath))
    
    return removed_files

# Usage
removed = cleanup_old_cache_files(days_old=7)
print(f"Removed {len(removed)} old cache files: {removed}")
```

## 🔧 Cache Management

### Automatic Cache Generation
```python
from core.embeddings import generate_embeddings

# This will automatically cache embeddings if not present
embeddings, languages = generate_embeddings("dog", "LaBSE")
```

### Manual Cache Creation
```python
from core.embeddings import cache_embeddings
import numpy as np

# Create embeddings manually
embeddings = np.random.randn(10, 768)  # 10 languages, 768 dimensions
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Cache embeddings
cache_path = cache_embeddings(embeddings, "test_concept", "LaBSE")
print(f"Cached embeddings at: {cache_path}")
```

### Cache Invalidation
```python
import os

def invalidate_cache(concept_name, encoder="LaBSE"):
    """Remove cache file to force regeneration."""
    cache_path = f"data/embeddings/{concept_name}_{encoder}_cached.npy"
    
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Invalidated cache: {cache_path}")
    else:
        print(f"Cache not found: {cache_path}")

# Usage
invalidate_cache("dog", "LaBSE")
```

## 📊 Performance Characteristics

### Cache Performance
- **Load time**: ~10ms for typical embeddings
- **Memory usage**: 75-225 KB per file
- **Storage efficiency**: Compressed NumPy format
- **Access speed**: Instant loading from disk

### Generation Performance
- **First-time generation**: 30-60 seconds per concept
- **Subsequent access**: Instant from cache
- **Memory usage**: <1GB during generation
- **Network usage**: Downloads encoder models once

## 🧪 Testing

Cache files have comprehensive validation:

- **Format validation**: NumPy array structure
- **Content validation**: Embedding quality and normalization
- **Integration testing**: End-to-end caching workflows
- **Performance testing**: Load time and memory usage

## 📚 References

- **NumPy**: Numerical array storage
- **LaBSE**: Language-agnostic BERT embeddings
- **Sentence Transformers**: Multilingual embedding models
- **Caching**: Performance optimization strategy 