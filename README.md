# Semantic Ising Simulator (Alpha Ver.)

A multilingual semantic Ising model simulator that explores how semantically identical words across languages potentially converge in their embedding space under Ising dynamics. 
The simulator visualizes multilingual embedding alignment and reveals latent structure as the language space approaches critical temperature.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Scientific Background](#scientific-background)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Future Work](#future-work)

## 🎯 Overview

This simulator investigates the **Platonic representation hypothesis**: do words meaning "dog" in 70+ languages share a common latent semantic structure? Using Ising model dynamics, we simulate semantic phase transitions and detect critical temperatures where universal semantic patterns emerge.

**Key Questions:**
- Do semantically identical words across languages converge in embedding space?
- What is the critical temperature where semantic phase transitions occur?
- How do anchor languages relate to emergent multilingual semantic structures?

## ✨ Features

- ** Multilingual Support**: 70+ languages with LaBSE embeddings
- ** Ising Dynamics**: Metropolis/Glauber update rules with temperature sweeps
- ** Critical Temperature Detection**: log(ξ) derivative method for phase transition detection
- ** Anchor Language Analysis**: Single-phase vs two-phase experimental designs
- ** Interactive Visualizations**: UMAP projections, entropy curves, correlation analysis
- ** Advanced Metrics**: Cosine distance and similarity for anchor comparison
- **🖥 Streamlit UI**: User-friendly interface with real-time simulation monitoring

## 📝 Recent Improvements
- UI and backend are now tightly synchronized for critical temperature (Tc) display: the value shown in the UI and the vertical line in charts are always consistent.
- The convergence summary chart now only shows the vertical Tc line, improving clarity.
- Anchor comparison and simulation tabs have been cleaned up for a more professional, user-friendly experience.
- All debug output has been removed from the user interface.

## Screenshots Sneakpeak

### Visualize multilingual concept mappings
<img src="https://github.com/user-attachments/assets/29af9400-1be4-49c2-92b5-cd2925a001ac" width="100%"/>
<br>

### Identify critical convergences across languages from alignment simulations
<img src="https://github.com/user-attachments/assets/4ed4c23b-0814-427f-b271-32959b6bb7e6" width="100%"/>
<br>

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Simulator

```bash
streamlit run app.py
```

### 3. Configure Your Experiment

1. **Select Concept**: Choose a concept (e.g., "dog", "tree", "house")
2. **Set Temperature Range**: Use auto-estimate or set manually (0.1-5.0)
3. **Configure Anchor**: Choose anchor language and include/exclude from dynamics
4. **Run Simulation**: Click "Run Simulation" and watch the magic happen!

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

## 🎮 Usage

### Web Interface

1. **Overview Tab**: Learn about the simulator and scientific background
2. **Simulation Results**: Run experiments and view real-time metrics
3. **Anchor Comparison**: Analyze anchor language relationships

### Command Line

```bash
# Run simulation with custom parameters
python main.py --concept dog --encoder LaBSE --t-min 0.1 --t-max 3.0 --t-steps 50

# Use configuration file
python main.py --config my_experiment.yaml
```

### Experimental Designs

Two modes to study multilingual semantic structure:

#### 🔬 **Single-Phase Mode** (`include_anchor=True`)
**Question**: "Does the anchor language share semantic space with other languages?"
- Anchor participates in Ising dynamics with all languages
- **Use when**: You want to see how anchor influences collective dynamics

#### 🔬 **Two-Phase Mode** (`include_anchor=False`)
**Question**: "How does the anchor compare to the emergent multilingual structure?"
- Anchor excluded from dynamics, compared to result at critical temperature
- **Use when**: You want to test anchor alignment with emergent structure

#### 📊 **Key Differences**
- **Single-Phase**: Higher Tc, anchor visible in UMAP
- **Two-Phase**: Lower Tc, anchor highlighted separately in UMAP

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

## 📁 Project Structure

```
semantic-ising/
├── app.py                 # Main Streamlit application
├── main.py               # CLI interface
├── config/               # Configuration management
│   ├── defaults.yaml     # Default parameters
│   └── validator.py      # Config validation
├── core/                 # Core simulation engine
│   ├── simulation.py     # Temperature sweeps & Ising updates
│   ├── embeddings.py     # Multilingual embedding pipeline
│   ├── phase_detection.py # Critical temperature detection
│   ├── post_analysis.py  # Post-simulation analysis
│   └── ...               # Other core modules
├── ui/                   # User interface components
│   ├── charts.py         # Interactive visualizations
│   ├── components.py     # Reusable UI components
│   └── tabs/             # Tab-specific components
├── data/                 # Data and embeddings
│   ├── concepts/         # Multilingual concept files
│   └── embeddings/       # Cached embeddings
├── tests/                # Comprehensive test suite
└── export/               # Export and I/O utilities
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with test coverage
pytest --cov=core --cov=ui tests/
```

### Key Areas for Contribution

- **New Concepts**: Add multilingual concept files in `data/concepts/`
- **Visualizations**: Enhance chart functions in `ui/charts.py`
- **Metrics**: Implement new comparison metrics in `core/comparison_metrics.py`
- **Documentation**: Improve scientific documentation and tutorials

## 🔮 Future Work

We have exciting plans to expand the Semantic Ising Simulator's capabilities:

### 🌐 Enhanced Multilingual Support
- **XLM-R Integration**: Enable XLM-RoBERTa embeddings for improved cross-lingual performance
- **mBERT Functionality**: Add multilingual BERT support for broader model comparison
- **Model Comparison**: Compare semantic dynamics across different embedding architectures

### 📊 Multi-Concept Comparisons
- **Concept Comparison Mode**: Compare up to 3 word concepts simultaneously
- **Flexible Selection**: Choose 1 concept or compare 1-2 concepts based on research needs
- **Enhanced Visualizations**: Extend existing charts to show multiple concepts:
  - Side-by-side alignment curves
  - Comparative UMAP projections
  - Cross-concept correlation analysis
  - Critical temperature comparisons

### 🎯 Research Applications
- **Semantic Field Analysis**: Study relationships between related concepts (e.g., "dog" vs "cat" vs "animal")
- **Cross-Domain Comparisons**: Compare concrete vs abstract concepts across languages
- **Temporal Evolution**: Track semantic changes across different embedding model versions

### 🔧 Technical Enhancements
- **Performance Optimization**: GPU acceleration for large-scale simulations
- **Real-time Collaboration**: Multi-user simulation sessions
- **Advanced Analytics**: Machine learning-based pattern detection in semantic dynamics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For LaBSE multilingual embeddings
- **Streamlit**: For the interactive web interface
- **Plotly**: For interactive visualizations
- **UMAP**: For dimensionality reduction and clustering

## 📚 References

- [LaBSE: Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852)
- [Ising Model and Phase Transitions](https://en.wikipedia.org/wiki/Ising_model)
- [Correlation Length in Critical Phenomena](https://en.wikipedia.org/wiki/Correlation_length)

---

**Ready to discover universal semantic structures?** 🚀

[Get Started](#quick-start) | [View Examples](examples/) | [Report Issues](https://github.com/pixiiidust/semantic-ising/issues) | [Discussions](https://github.com/pixiiidust/semantic-ising/discussions)
