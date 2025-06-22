# Semantic Ising Simulator (Alpha Ver.)

A multilingual semantic Ising model simulator that: 
1. Explores how semantically identical words across languages potentially converge in their embedding space (under Ising dynamics). 
2. Visualizes multilingual alignments to reveal latent structure as the system approaches a critical threshold.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Experimental Designs](#-experimental-designs)
- [Scientific Background](#-scientific-background)
- [Recent Improvements](#-recent-improvements)
- [Contributing](#-contributing)
- [References](#-references)

---

## 🎯 Overview

Do words meaning "dog" in 70+ languages share a common latent semantic structure? 
* Using Ising model dynamics, we simulate semantic phase transitions by detecting critical temperatures
* Critical temperatures denote alignment thresholds for embedded multilingual spaces, where universal semantic patterns may emerge
* This speculative approach is inspired by the **Platonic representation hypothesis** 

**Key Research Questions:**
1. Do semantically identical words across languages converge towards a universal embedding space?
2. What is the critical temperature where semantic phase transitions occur?
3. How do anchor languages relate to emergent multilingual semantic structures?

For a deeper dive and visualizing the concepts behind this simulator, see the [Scientific Background](#-scientific-background) section.

---

## ✨ Features

### 🔬 Core Simulation
- **Multilingual Support**: 70+ languages with LaBSE embeddings
- **Ising Dynamics**: Metropolis/Glauber update rules with temperature sweeps
- **Disk-based Snapshot Storage**: Persistent storage of simulation vectors at each temperature step

### 📊 Analysis & Visualization
- **Interactive UMAP Visualization**: Temperature slider for dynamic exploration of semantic structure
- **Critical Temperature Detection**: log(ξ) derivative method for phase transition detection
- **Advanced Metrics**: Cosine distance and similarity for anchor comparison

### 🖥️ User Interface
- **Streamlit UI**: User-friendly interface with real-time simulation monitoring
- **Interactive Temperature Control**: Temperature slider for dynamic UMAP visualization
- **Enhanced Metrics Display**: Three-column layout showing Critical Temperature, Cosine Distance, and Cosine Similarity

---

## 📸 Screenshots

### Visualize multilingual concept mappings
<img src="https://github.com/user-attachments/assets/3c0ce8c2-2c13-487e-b923-817150e912ec"/>
### Visualize multilingual concept mappings
<img src="https://github.com/user-attachments/assets/3c0ce8c2-2c13-487e-b923-817150e912ec"/>

---

## 🚀 Quick Start

### 0. (Recommended) Create a Python Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

See [SETUP.md](SETUP.md) for detailed instructions and troubleshooting.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

##

### 2. Run the Simulator Dashboard

```bash
streamlit run app.py
```

##

### 3. Configure Your Experiment

#### 📄 Ensure Correct File Formats

##### **JSON Structure Example:**
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
**File Naming Convention:**

- **Standard format**: `{concept}_translations_25.json` (25 languages)
- **Extended format**: `{concept}_translations_75.json` (75 languages)

**File Properties:**

- **Encoding**: UTF-8
- **Format**: Valid JSON
- **Language codes**: ISO 639-1 standard
- **Translations**: Single words or short phrases

**Important Limitation:**

- **Current version only supports the same concept across different languages**
- Each JSON file must contain translations of the **same semantic concept** (e.g., all words meaning "dog")
- **Do not mix different concepts** in the same file (e.g., mixing "dog" and "tree" translations)
- The system assumes all translations in a file are semantically equivalent for proper Ising dynamics analysis

##

### 4. Usage Steps

1. **Select Concept**: Choose a concept (e.g., "dog", "tree", "love") or upload your own concepts
2. **Set Temperature Range**: Use auto-estimate (recommended) or set manually (0.1-5.0)
3. **Configure Anchor**: Choose anchor language and include/exclude from dynamics
4. **Run Simulation**: Click "Run Simulation" and watch the magic happen!

---

## 📦 Installation

For detailed setup instructions, system requirements, implementation details, and technical documentation, please see our [Setup Guide](SETUP.md). For the canonical project structure and module dependencies, refer to [directory_structure.lua](directory_structure.lua).

### Quick Install

```bash
# Clone and install
git clone https://github.com/pixiiidust/semantic-ising.git
cd semantic-ising
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Alternative: Docker

```bash
# Build and run with Docker
docker build -t semantic-ising .
docker run -p 8501:8501 semantic-ising
```

See [SETUP.md](SETUP.md) for complete installation options, dependencies, and system requirements.

---

## 🎮 Usage

### Web Interface (Dashboard)

1. **Overview Tab**: Learn about the simulator and scientific background
2. **Simulation Results**: Output from simulations for viewing metrics
3. **Anchor Comparison**: Analyze anchor language relationships
4. **Sidebar**: Configuration settings

### Command Line Interface

```bash
# Run simulation with custom parameters
python main.py --concept dog --encoder LaBSE --t-min 0.1 --t-max 3.0 --t-steps 50

# Use configuration file
python main.py --config my_experiment.yaml
```

---

## 🔬 Experimental Designs

Two modes to study multilingual semantic structure:

### **Single-Phase Mode** (`include_anchor=True`)

**Question**: "Does the anchor language share semantic space with other languages?"
- Anchor participates in Ising dynamics with all languages
- **Use when**: You want to see how anchor influences collective dynamics

### **Two-Phase Mode** (`include_anchor=False`)

**Question**: "How does the anchor compare to the emergent multilingual structure?"
- Anchor excluded from dynamics, compared to result at critical temperature
- **Use when**: You want to test anchor alignment with emergent structure

#### 📊 **Key Differences**

- **Single-Phase**: Higher Tc, anchor visible in UMAP
- **Two-Phase**: Lower Tc, anchor highlighted separately in UMAP

---

## 🧠 Scientific Background

### The Ising Model

For an intuitive visualization of spin alignments in an Ising model, click to watch this educational clip:

<video width="630" height="300" src="https://github.com/user-attachments/assets/c22d18dc-3e56-4713-aae5-c5a6e8cc48fb"></video>

*Video Source: [@F_Sacco](https://youtu.be/cGcY-ReeGDU?) / [francesco215.github.io (Requirements for self organization)](https://francesco215.github.io/Language_CA/)*

### Adapting the Ising Model for Semantics

This tool applies a continuous, semantic variant of the Ising model using multilingual concept embeddings:

* **Vectors as Spins**: Embeddings (e.g. 768D LaBSE vectors) act as "spins," aligning or misaligning in semantic space.
* **Continuous Updates**: `update_vectors_metropolis()` and `update_vectors_glauber()` perturb vectors with Gaussian noise, accepting changes based on energy shifts.
* **Semantic Alignment**: Updates reflect meaning shifts—vectors move closer or farther in semantic space.
* **Temperature Control**: Higher temperature (T) increases randomness; lower T encourages alignment.
* **Phase Transitions**: At a critical temperature (Tc), global structure emerges—mirroring phase transitions.
* **Correlation Length**: Measures the scale of semantic coherence across the system.

---

### Key Metrics

- **Alignment**: Average cosine similarity between concept vectors (0-1 scale)
- **Entropy**: Shannon entropy of vector distribution
- **Correlation Length**: Characteristic length scale of correlations
- **Cosine Distance**: Primary semantic distance metric for anchor comparison (0-1, lower is better)
- **Cosine Similarity**: Directional similarity for anchor comparison (0-1, higher is better)

### Phase Transition Detection

- The simulator detects the critical temperature (Tc) using the log(ξ) derivative method.
- Tc is estimated by identifying temperature where the correlation length (ξ) collapses (the "knee" in the plot).
- This provides a robust and physically meaningful detection of phase transitions.

---

## 📝 Recent Improvements

- **Interactive UMAP Visualization**: Added temperature slider for dynamic exploration of semantic structure across temperature steps
- **Disk-based Snapshot Storage**: Implemented persistent storage of simulation vectors for memory efficiency and large simulation support
- **Language Code Preservation**: Fixed UMAP language labels to show actual language codes (en, es, fr, etc.) instead of generic labels
- **Enhanced Metrics Display**: Three-column layout with Critical Temperature, Cosine Distance, and Cosine Similarity prominently displayed
- **Memory Optimization**: Temperature-based snapshot loading reduces memory usage for large simulations
- **Anchor Comparison Metrics Fix**: Resolved inconsistency between main comparison metrics and interactive metrics by ensuring both use original anchor vectors for comparison
- **Disk-based Recalculation**: Fixed anchor comparison recalculation to properly handle disk-based snapshots when in-memory snapshots are not available
- **Consistent Metrics Display**: All UI components now display consistent cosine similarity and distance metrics at the critical temperature
- **Technical Debt Reduction**: Removed outdated references to unimplemented features (mBERT, XLM-R, Binder cumulant method, linguistic distance weighting)

---

## 🤝 Contributing

We welcome contributions! Before contributing:

1. Review our [Setup Guide](SETUP.md) for detailed implementation and technical documentation
2. Check [directory_structure.lua](directory_structure.lua) for the canonical project structure, which defines:
   - Complete module hierarchy
   - File dependencies
   - Test coverage requirements
   - Component relationships
3. Ensure your changes align with our project architecture and coding standards

See our [Contributing Guidelines](CONTRIBUTING.md) for detailed instructions.

---

## 📚 References

- [LaBSE: Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852)
- [The Ising model celebrates a century of interdisciplinary contributions](https://www.nature.com/articles/s44260-024-00012-0)
- [Correlation Length in Critical Phenomena](https://en.wikipedia.org/wiki/Correlation_length)
- [Sacco, et al., "Requirements for self organization", zenodo, 2023](https://zenodo.org/records/8416764)

---

**Ready to discover universal semantic structures?** 🚀

[Report Issues](https://github.com/pixiiidust/semantic-ising/issues) | [Discussions](https://github.com/pixiiidust/semantic-ising/discussions)
