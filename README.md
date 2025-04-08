# Zero-Cost Neural Architecture Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

A Python package for efficient neural architecture search without training, based on zero-cost proxies.

## Overview

Zero-Cost Neural Architecture Search (ZC-NAS) is a technique that allows finding good neural network architectures without the computational expense of training models. This package implements several zero-cost proxies from recent literature and combines them into an ensemble metric for more reliable architecture evaluation.

## Paper Review

The concept of zero-cost neural architecture search was introduced in the paper "Neural Architecture Search without Training" by Mellor et al. (2021). The authors proposed a correlation-based metric that can predict the performance of neural network architectures by analyzing the correlation between activations of different data samples.

Subsequent works have expanded on this idea:

- "Zero-Cost Proxies for Lightweight NAS" (Abdelfattah et al., 2021) introduced gradient-based metrics that measure the alignment of gradients across different samples.
- "ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients" (Li et al., 2022) proposed a metric based on the inverse coefficient of variation of gradients.
- "Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets" (Yan et al., 2022) explored dataset-aware architecture prediction.

This package implements these metrics and combines them into an ensemble for more robust architecture evaluation.

## Method

My approach uses an ensemble of zero-cost proxies to evaluate neural network architectures without training. The ensemble score is a weighted combination of individual metrics such that, for an architecture $A$ on dataset $D$, is computed as

$$S_{ensemble}(A, D) = \sum_{i=1}^{n} w_i \cdot \text{normalize}(m_i(A, D))$$

where:
- $m_i(A, D)$ is the $i$-th metric evaluated on architecture $A$ and dataset $D$
- $\text{normalize}(\cdot)$ scales the metric to the range $[0, 1]$
- $w_i$ is the weight assigned to the $i$-th metric
- $n$ is the number of metrics in the ensemble

The metrics we use include:

1. **Activation Correlation** (Mellor et al., 2021): Measures the correlation between activations of different samples.
2. **Gradient Conflict** (Abdelfattah et al., 2021): Measures the alignment of gradients across different samples.
3. **ZiCo** (Li et al., 2022): Measures the inverse coefficient of variation of gradients.
4. **Synflow** (Tanaka et al., 2020): Measures the product of parameters and gradients.
5. **GraSP** (Wang et al., 2020): Measures the gradient signal preservation.

## Installation

```bash
pip install zcs
```

Or install from source:

```bash
git clone https://github.com/yourusername/zero_cost_search.git
cd zcs
pip install -e .
```

## Usage

### Command-Line Interface

The package provides a command-line interface for easy use:

```bash
# Basic usage
zcs --data your_data.pt --output results.json

# With visualization
zcs --data your_data.pt --output results.json --plot search_results.png

# Customizing the search space
zcs --data your_data.pt \
                 --depths 2 3 4 \
                 --widths 64 128 256 \
                 --activations relu tanh leaky_relu \
                 --num-samples 20

# Help
zcs --help
```

### Python API

### Basic Usage

```python
import torch
from zcs import ZeroCostNAS

# Prepare your data
X = torch.randn(100, 20)  # 100 samples, 20 features
y = torch.randint(0, 3, (100,))  # 3 classes

# Initialize the NAS
nas = ZeroCostNAS(
    input_dim=20,  # Input dimension
    output_dim=3,  # Output dimension (number of classes)
    seed=42  # For reproducibility
)

# Quick prediction using meta-learning
predicted_config = nas.predict_architecture(X, y)
print(f"Predicted architecture: {predicted_config}")

# Full search with zero-cost proxies
result = nas.search(
    X=X,
    y=y,
    depths=[2, 3, 4],  # Network depths to consider
    widths=[64, 128, 256],  # Layer widths to consider
    activations=['relu', 'tanh', 'leaky_relu'],  # Activation functions
    num_samples=20,  # Number of architectures to evaluate
    use_meta_learning=True  # Use meta-learning to guide the search
)

# Get the best model
best_model = nas.get_best_model(include_dropout=0.2, include_batchnorm=True)
print(f"Best model: {best_model}")

# Get detailed results
summary = nas.get_results_summary()
print(f"Best configuration: {summary['best_config']}")
print(f"Best score: {summary['best_score']:.4f}")
```

### Visualization

```python
from zcs import plot_search_results

# Plot the search results
plot_search_results(result, save_path="search_results.png")
```

### Logging

```python
from zcs import setup_logger

# Set up logging
logger = setup_logger(level="INFO", log_file="zcs.log")
```

## Contributing

Contributions are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your project, please cite it using

```bibtex
@software{sadoune2025zerocost,
  title={Zero-Cost-Search: A Python Package for Zero-Cost Neural Architecture Search},
  author={Sadoune, Igor},
  year={2025},
  url={https://github.com/igorsadoune/zcs}
}
```