# Contributing to Semantic Ising Simulator

We welcome contributions! This document provides guidelines for contributing to the Semantic Ising Simulator project.

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