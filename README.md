# Semantic Ising Simulator (Alpha Ver.)

A multilingual semantic Ising model simulator that: 
1. Explores how semantically identical words across languages potentially converge in their embedding space (under Ising dynamics). 
2. Visualizes multilingual alignments to reveal latent structure as the system approaches a critical threshold.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Designs](#experimental-designs)
- [Scientific Background](#scientific-background)
- [Recent Improvements](#recent-improvements)
- [Contributing](#contributing)

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

---

## ✨ Features

### 🔬 Core Simulation
- **Multilingual Support**: 70+ languages with LaBSE embeddings
- **Ising Dynamics**: Metropolis/Glauber update rules with temperature sweeps
- **Critical Temperature Detection**: log(ξ) derivative method for phase transition detection
- **Smart Temperature Estimation**: Auto-estimation with configurable limits and conservative energy scaling

### 📊 Analysis & Visualization
- **Anchor Language Analysis**: Single-phase vs two-phase experimental designs
- **Interactive Visualizations**: UMAP projections, entropy curves, correlation analysis
- **Advanced Metrics**: Cosine distance and similarity for anchor comparison

### 🖥️ User Interface
- **Streamlit UI**: User-friendly interface with real-time simulation monitoring and progress tracking

---

## 📸 Screenshots

### Visualize multilingual concept mappings
<img src="https://github.com/user-attachments/assets/29af9400-1be4-49c2-92b5-cd2925a001ac" width="100%"/>

### Identify critical convergences across languages from alignment simulations
<img src="https://github.com/user-attachments/assets/4ed4c23b-0814-427f-b271-32959b6bb7e6" width="100%"/>

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Simulator Dashboard

```bash
streamlit run app.py
```

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
  ##### **File Naming Convention:**
- **Standard format**: `{concept}_translations_25.json` (25 languages)
- **Extended format**: `{concept}_translations_75.json` (75 languages)

  ##### **File Properties:**
- **Encoding**: UTF-8
- **Format**: Valid JSON
- **Language codes**: ISO 639-1 standard
- **Translations**: Single words or short phrases

  ##### **⚠️ Important Limitation:**
- **Current version only supports the same concept across different languages**
- Each JSON file must contain translations of the **same semantic concept** (e.g., all words meaning "dog")
- **Do not mix different concepts** in the same file (e.g., mixing "dog" and "tree" translations)
- The system assumes all translations in a file are semantically equivalent for proper Ising dynamics analysis

### 4. Usage Steps

1. **Select Concept**: Choose a concept (e.g., "dog", "tree", "love") or upload your own concepts
2. **Set Temperature Range**: Use auto-estimate (recommended) or set manually (0.1-5.0)
3. **Configure Anchor**: Choose anchor language and include/exclude from dynamics
4. **Run Simulation**: Click "Run Simulation" and watch the magic happen!

---

## 📦 Installation

### Prerequisites

- Python 3.11 or 3.10 (recommended for PyTorch compatibility)
- 4GB+ RAM for large language sets

### Install

```bash
# Clone the repository
git clone https://github.com/pixiiidust/semantic-ising.git
cd semantic-ising

# Install dependencies
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

### Ising Model for Semantics

The simulator applies statistical physics concepts to semantic analysis:

- **Vectors as Spins**: Word embeddings represent "spins" in semantic space
- **Temperature Control**: Higher T = more randomness, Lower T = more alignment
- **Phase Transitions**: Critical temperature (Tc) marks emergence of universal structure
- **Correlation Length**: Characteristic scale of semantic correlations

### Key Metrics

- **Alignment**: Average cosine similarity between vectors (0-1 scale)
- **Entropy**: Shannon entropy of vector distribution
- **Correlation Length**: Characteristic length scale of correlations
- **Cosine Distance**: Primary semantic distance metric for anchor comparison (0-1, lower is better)
- **Cosine Similarity**: Directional similarity for anchor comparison (0-1, higher is better)

### Phase Transition Detection

- The simulator now detects the critical temperature (Tc) using the log(ξ) derivative method: Tc is identified as the temperature where the correlation length (ξ) collapses (the "knee" in the plot). This is more robust and physically meaningful than previous alignment-based or Binder cumulant methods.

- All UI and analysis now annotate Tc at the knee, matching physical expectations.

---

## 📝 Recent Improvements

- **Temperature Estimation**: Enhanced auto-estimation with configurable max temperature limits and conservative 2.0× energy fluctuation multiplier
- **UI Streamlining**: Removed non-functional power law analysis tab from simulation results for cleaner interface
- **Progress Tracking**: Real-time progress bars and status updates during temperature sweeps
- **Config Integration**: Temperature estimation now respects maximum temperature settings from config files
- **UI and backend synchronization**: Critical temperature (Tc) display is consistent between UI and charts
- **Convergence Analysis**: Enhanced convergence summary with entropy vs correlation length visualization
- **Professional UI**: Cleaner, more user-friendly interface with improved explanations and removed debug output

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

---

**Ready to discover universal semantic structures?** 🚀

[Report Issues](https://github.com/pixiiidust/semantic-ising/issues) | [Discussions](https://github.com/pixiiidust/semantic-ising/discussions)
