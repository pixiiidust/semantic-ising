# Data Module Documentation

The `data/` directory contains multilingual concept files and cached embeddings for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Concepts](#concepts)
- [Embeddings](#embeddings)
- [File Formats](#file-formats)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The data directory serves as the multilingual knowledge base for the simulator, providing:

- **Multilingual concept translations** in JSON format
- **Cached embeddings** to avoid recomputation
- **Structured data organization** for easy access
- **Version control** for different language sets

## 📁 Directory Structure

```
data/
├── concepts/           # Multilingual concept translations
│   ├── dog_translations_25.json
│   ├── dog_translations_75.json
│   ├── tree_translations_25.json
│   ├── tree_translations_75.json
│   ├── i_love_you_translations_25.json
│   ├── i_love_you_translations_75.json
│   ├── love_translations_25.json
│   ├── love_translations_75.json
│   └── concepts.md
└── embeddings/         # Cached embeddings
    ├── dog_LaBSE_cached.npy
    ├── dog_LaBSE_dog_translations_LaBSE_cached.npy
    ├── dog_LaBSE_dog_translations_72_LaBSE_cached.npy
    ├── dog_25.json_LaBSE_dog_translations_25_LaBSE_cached.npy
    ├── tree_LaBSE_tree_translations_LaBSE_cached.npy
    ├── tree_LaBSE_tree_translations_72_LaBSE_cached.npy
    ├── i_love_you_25.json_LaBSE_i_love_you_translations_25_LaBSE_cached.npy
    ├── i_love_you_75.json_LaBSE_i_love_you_translations_75_LaBSE_cached.npy
    ├── love_LaBSE_love_translations_75_LaBSE_cached.npy
    └── embeddings.md
```

## 🔤 Concepts

### File Naming Convention
- **Standard format**: `{concept}_translations_25.json` (25 languages)
- **Extended format**: `{concept}_translations_75.json` (75 languages)

### Supported Concepts
- **dog**: Canine animal translations (25 & 75 languages)
- **tree**: Plant/tree translations (25 & 75 languages)
- **i_love_you**: Universal expression of love (25 & 75 languages)
- **love**: Universal concept of love (25 & 75 languages)
- **house**: Building/home translations (planned)
- **car**: Vehicle translations (planned)

### Language Coverage
- **Standard sets**: 25 languages (common languages)
- **Extended sets**: 75 languages (comprehensive coverage)

### File Structure
```json
{
  "en": "dog",
  "es": "perro", 
  "fr": "chien",
  "de": "hund",
  "it": "cane",
  "pt": "cachorro",
  "ru": "собака",
  "zh": "狗",
  "ja": "犬",
  "ko": "개"
}
```

### Available Concept Files

#### Standard Sets (25 languages)
- `dog_translations_25.json` - Dog translations in common languages
- `tree_translations_25.json` - Tree translations in common languages
- `i_love_you_translations_25.json` - I love you translations in common languages
- `love_translations_25.json` - Love translations in common languages

#### Extended Sets (75 languages)
- `dog_translations_75.json` - Dog translations in 75 languages
- `tree_translations_75.json` - Tree translations in 75 languages
- `i_love_you_translations_75.json` - I love you translations in 75 languages
- `love_translations_75.json` - Love translations in 75 languages

## 🧠 Embeddings

### Caching Strategy
- **Automatic caching**: Embeddings generated and cached automatically
- **Cache validation**: Corrupted cache files detected and regenerated
- **Version control**: Different cache files for different language sets

### File Naming Convention
- **Format**: `{concept}_{encoder}_{concept_file}_{encoder}_cached.npy`
- **Example**: `dog_LaBSE_dog_translations_72_LaBSE_cached.npy`

### Supported Encoders
- **LaBSE**: Language-agnostic BERT Sentence Embedding (primary)
- **SBERT**: Sentence-BERT (planned)
- **XLM-R**: XLM-RoBERTa (planned)

### Cache File Properties
- **Format**: NumPy arrays (.npy)
- **Shape**: (n_languages, embedding_dimension)
- **Normalization**: Unit-length vectors
- **Data type**: float32 for efficiency

### Available Cache Files
- `dog_LaBSE_cached.npy` - Basic dog embeddings (LaBSE)
- `dog_LaBSE_dog_translations_LaBSE_cached.npy` - Basic dog embeddings
- `dog_LaBSE_dog_translations_72_LaBSE_cached.npy` - Extended dog embeddings (72 languages)
- `tree_LaBSE_tree_translations_LaBSE_cached.npy` - Basic tree embeddings (LaBSE)
- `tree_LaBSE_tree_translations_72_LaBSE_cached.npy` - Extended tree embeddings (72 languages)
- `i_love_you_25.json_LaBSE_i_love_you_translations_25_LaBSE_cached.npy` - Extended i_love_you embeddings (25 languages)
- `i_love_you_75.json_LaBSE_i_love_you_translations_75_LaBSE_cached.npy` - Extended i_love_you embeddings (75 languages)
- `love_LaBSE_love_translations_75_LaBSE_cached.npy` - Extended love embeddings (75 languages)

## 📄 File Formats

### JSON Concept Files
```json
{
  "language_code": "translation",
  "en": "dog",
  "es": "perro",
  "fr": "chien"
}
```

**Properties**:
- UTF-8 encoding
- Valid JSON format
- Language codes follow ISO 639-1 standard
- Translations are single words or short phrases

### NumPy Cache Files
```python
import numpy as np

# Load cached embeddings
embeddings = np.load("data/embeddings/dog_LaBSE_cached.npy")

# Properties
print(f"Shape: {embeddings.shape}")  # (n_languages, embedding_dim)
print(f"Data type: {embeddings.dtype}")  # float32
print(f"Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")  # True
```

## 💡 Usage Examples

### Loading Concept Translations
```python
import json
import os

def load_concept_translations(concept_name):
    """Load concept translations from JSON file."""
    filepath = f"data/concepts/{concept_name}_translations_25.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    
    return translations

# Usage
translations = load_concept_translations("dog")
print(f"Languages: {list(translations.keys())}")
print(f"English: {translations['en']}")
print(f"Spanish: {translations['es']}")
```

### Loading Cached Embeddings
```python
import numpy as np

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

### Adding New Concepts
```python
import json

def create_concept_file(concept_name, translations):
    """Create a new concept translation file."""
    filepath = f"data/concepts/{concept_name}_translations_25.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    
    print(f"Created concept file: {filepath}")

# Example: Create cat concept
cat_translations = {
    "en": "cat",
    "es": "gato",
    "fr": "chat",
    "de": "katze",
    "it": "gatto",
    "pt": "gato",
    "ru": "кот",
    "zh": "猫",
    "ja": "猫",
    "ko": "고양이"
}

create_concept_file("cat", cat_translations)
```

## 🔧 Data Management

### Cache Management
- **Automatic cleanup**: Old cache files can be removed
- **Regeneration**: Corrupted files automatically regenerated
- **Version control**: Different versions for different language sets

### Quality Control
- **Translation validation**: Ensure accurate translations
- **Language code validation**: Verify ISO 639-1 compliance
- **Encoding validation**: Ensure proper UTF-8 encoding

### Performance Optimization
- **Caching**: Avoid expensive embedding recomputation
- **Compression**: Efficient storage of large embedding arrays
- **Lazy loading**: Load data only when needed

## 🌍 Language Support

### Basic Language Set (25 languages)
- **European**: English, Spanish, French, German, Italian, Portuguese, Russian
- **Asian**: Chinese, Japanese, Korean
- **Other**: Arabic, Hindi, Turkish

### Extended Language Set (75 languages)
- **Comprehensive coverage** of major world languages
- **Regional variations** and dialects
- **Low-resource languages** for research purposes

### Language Code Standards
- **ISO 639-1**: Two-letter language codes (en, es, fr)
- **ISO 639-2**: Three-letter codes for extended languages
- **Consistent mapping** across all concept files

## 📊 Data Statistics

### Current Coverage
- **Concepts**: 5 (dog, tree, i_love_you, house, car)
- **Language sets**: 2 (standard, extended)
- **Total translations**: 200+ unique translations
- **Cache files**: 6 embedding cache files

### File Sizes
- **Concept files**: 1-2 KB each
- **Cache files**: 75-225 KB each
- **Total data size**: ~1 MB

## 🧪 Testing

Data files have comprehensive validation:

- **Format validation**: JSON structure and encoding
- **Content validation**: Translation accuracy and completeness
- **Cache validation**: Embedding quality and normalization
- **Integration testing**: End-to-end data loading workflows

## 📚 References

- **ISO 639-1**: Language code standards
- **LaBSE**: Language-agnostic BERT embeddings
- **NumPy**: Numerical array storage
- **JSON**: Data interchange format 