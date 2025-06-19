# Concepts Directory Documentation

The `data/concepts/` directory contains multilingual concept translation files for the Semantic Ising Simulator.

## 📋 Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [Supported Concepts](#supported-concepts)
- [Language Coverage](#language-coverage)
- [File Formats](#file-formats)
- [Usage Examples](#usage-examples)

## 🎯 Overview

The concepts directory serves as the multilingual knowledge base for semantic experiments, providing:

- **Multilingual concept translations** in JSON format
- **Structured language mappings** for semantic analysis
- **Version control** for different language sets
- **Quality validation** for translation accuracy

## 📁 File Structure

```
data/concepts/
├── dog_translations_25.json        # Dog translations (25 languages)
├── dog_translations_75.json        # Extended dog translations (75 languages)
├── tree_translations_25.json       # Tree translations (25 languages)
├── tree_translations_75.json       # Extended tree translations (75 languages)
├── i_love_you_translations_25.json # I love you translations (25 languages)
├── i_love_you_translations_75.json # Extended I love you translations (75 languages)
├── love_translations_25.json       # Love translations (25 languages)
├── love_translations_75.json       # Extended love translations (75 languages)
└── concepts.md                     # This documentation file
```

## 🔤 Supported Concepts

### Current Concepts
- **dog**: Canine animal translations (25 & 75 languages)
- **tree**: Plant/tree translations (25 & 75 languages)
- **i_love_you**: Universal expression of love (25 & 75 languages)
- **love**: Universal concept of love (25 & 75 languages)

### Planned Concepts
- **house**: Building/home translations
- **car**: Vehicle translations
- **book**: Literature/reading translations
- **water**: Liquid/nature translations

## 🌍 Language Coverage

### Standard Language Sets (25 languages)
**European Languages**:
- English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Dutch (nl), Polish (pl), Russian (ru), Swedish (sv), Danish (da), Finnish (fi), Norwegian (no), Hungarian (hu), Czech (cs), Greek (el), Hebrew (he)

**Asian Languages**:
- Chinese (zh), Japanese (ja), Korean (ko), Thai (th), Vietnamese (vi)

**Other Languages**:
- Arabic (ar), Hindi (hi), Turkish (tr)

### Extended Language Sets (75 languages)
**Comprehensive coverage** including:
- **Indo-European**: English, Spanish, French, German, Italian, Portuguese, Russian, Hindi, Persian, Greek, etc.
- **Sino-Tibetan**: Chinese, Tibetan, Burmese, etc.
- **Afro-Asiatic**: Arabic, Hebrew, Amharic, etc.
- **Niger-Congo**: Swahili, Yoruba, Zulu, etc.
- **Austronesian**: Indonesian, Tagalog, Hawaiian, etc.
- **Uralic**: Finnish, Hungarian, Estonian, etc.
- **Caucasian**: Georgian, Armenian, Azerbaijani, etc.
- **And many more**...

## 📄 File Formats

### JSON Structure
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

### File Naming Convention
- **Standard format**: `{concept}_translations_25.json` (25 languages)
- **Extended format**: `{concept}_translations_75.json` (75 languages)

### File Properties
- **Encoding**: UTF-8
- **Format**: Valid JSON
- **Language codes**: ISO 639-1 standard
- **Translations**: Single words or short phrases

## 💡 Usage Examples

### Loading Concept Translations
```python
import json
import os

def load_concept_translations(concept_name):
    """Load concept translations from JSON file."""
    filepath = f"data/concepts/{concept_name}_translations.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    
    return translations

# Usage
translations = load_concept_translations("dog")
print(f"Languages: {list(translations.keys())}")
print(f"English: {translations['en']}")
print(f"Spanish: {translations['es']}")
```

### Available Concepts Detection
```python
import os
import json

def get_available_concepts():
    """Get list of available concepts."""
    concepts_dir = "data/concepts"
    concepts = []
    
    for filename in os.listdir(concepts_dir):
        if filename.endswith("_translations.json"):
            concept_name = filename.replace("_translations.json", "")
            if concept_name not in concepts:
                concepts.append(concept_name)
    
    return sorted(concepts)

# Usage
available_concepts = get_available_concepts()
print(f"Available concepts: {available_concepts}")
```

### Language Coverage Analysis
```python
def analyze_language_coverage(concept_name):
    """Analyze language coverage for a concept."""
    translations = load_concept_translations(concept_name)
    
    print(f"Concept: {concept_name}")
    print(f"Total languages: {len(translations)}")
    print(f"Language codes: {list(translations.keys())}")
    
    # Check for missing translations
    missing = []
    for lang_code, translation in translations.items():
        if not translation or translation.strip() == "":
            missing.append(lang_code)
    
    if missing:
        print(f"Missing translations: {missing}")
    else:
        print("All translations present")

# Usage
analyze_language_coverage("dog")
```

### Creating New Concepts
```python
def create_concept_file(concept_name, translations):
    """Create a new concept translation file."""
    filepath = f"data/concepts/{concept_name}_translations.json"
    
    # Validate translations
    if not isinstance(translations, dict):
        raise ValueError("Translations must be a dictionary")
    
    for lang_code, translation in translations.items():
        if not isinstance(lang_code, str) or len(lang_code) != 2:
            raise ValueError(f"Invalid language code: {lang_code}")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"Invalid translation for {lang_code}: {translation}")
    
    # Save to file
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

## 🔧 Quality Control

### Translation Validation
- **Accuracy**: Ensure translations are semantically equivalent
- **Consistency**: Maintain consistent terminology across concepts
- **Completeness**: Verify all language codes have translations
- **Encoding**: Ensure proper UTF-8 encoding for all scripts

### Language Code Standards
- **ISO 639-1**: Two-letter language codes (en, es, fr)
- **ISO 639-2**: Three-letter codes for extended languages
- **Consistent mapping**: Same codes used across all concept files

### File Validation
```python
def validate_concept_file(filepath):
    """Validate a concept translation file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        
        # Check structure
        if not isinstance(translations, dict):
            return False, "Root must be a dictionary"
        
        # Check language codes and translations
        for lang_code, translation in translations.items():
            if not isinstance(lang_code, str) or len(lang_code) != 2:
                return False, f"Invalid language code: {lang_code}"
            if not isinstance(translation, str) or not translation.strip():
                return False, f"Invalid translation for {lang_code}"
        
        return True, f"Valid file with {len(translations)} languages"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error: {e}"

# Usage
is_valid, message = validate_concept_file("data/concepts/dog_translations.json")
print(f"Validation: {message}")
```

## 📊 Statistics

### Current Coverage
- **Concepts**: 4 (dog, tree, i_love_you, love)
- **Language sets**: 2 (25 languages, 75 languages)
- **Total translations**: 450+ unique translations
- **File sizes**: 1-3 KB each

### Language Distribution
- **European**: 40% of languages
- **Asian**: 25% of languages
- **African**: 15% of languages
- **Other**: 20% of languages

## 🧪 Testing

Concept files have comprehensive validation:

- **Format validation**: JSON structure and encoding
- **Content validation**: Translation accuracy and completeness
- **Integration testing**: End-to-end concept loading workflows
- **Quality assurance**: Translation accuracy verification

## 📚 References

- **ISO 639-1**: Language code standards
- **JSON**: Data interchange format
- **UTF-8**: Unicode encoding standard
- **Multilingual NLP**: Natural language processing across languages 