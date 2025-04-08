# Zero-Cost Search Usage Guide

This document provides instructions on how to install, test, and use the Zero-Cost Search package.

## Installation

### Local Development Installation

1. Clone the repository (or download the package):
   ```bash
   git clone https://github.com/IgorSadoune/zcsearch.git
   cd zcsearch
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   Or use the Makefile:
   ```bash
   make install
   ```

3. Verify the installation:
   ```bash
   python test_local_install.py
   ```

### Installing from PyPI

```bash
pip install zcsearch
```

## Running Examples

### Simple Example

Run the simple example script:

```bash
python examples/run_example.py
```

Or use the Makefile:

```bash
make example
```

### Full Example

Run the full example with visualization:

```bash
python examples/example.py
```

Or use the Makefile:

```bash
make full-example
```

## Running Tests

Run the test suite:

```bash
pytest
```

Or use the Makefile:

```bash
make test
```

Run tests with coverage:

```bash
make test-cov
```

## Basic Usage in Your Project

```python
import torch
from zcsearch import ZeroCostNAS

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

# Use the model for inference
with torch.no_grad():
    outputs = best_model(X)
    predictions = outputs.argmax(dim=1)
```

## Logging and Visualization

```python
from zcsearch import setup_logger, plot_search_results

# Set up logging
logger = setup_logger(level="INFO", log_file="zcsearch.log")

# After running a search, visualize the results
plot_search_results(result, save_path="search_results.png")
```

## Advanced Usage

See the [README.md](README.md) for more detailed information on the package's features and capabilities.