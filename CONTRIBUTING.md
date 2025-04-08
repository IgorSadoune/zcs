# Contributing to Zero-Cost Search

Thank you for your interest in contributing to the Zero-Cost Search project! This document provides guidelines and instructions for contributing. However, if you are interested in contributing, please contact me at [igor.sadoune@pm.me](mailto:igor.sadoune@pm.me).

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Directory Structure](#directory-structure)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Git Workflow](#git-workflow)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Directory Structure

The project is organized as follows:

```
zero_cost_package/
├── zcsearch/                   # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── zero_cost_search.py     # Core implementation
│   ├── cli.py                  # Command-line interface
│   ├── models/                 # Neural network models
│   │   ├── __init__.py         # Models initialization
│   │   └── mlp.py              # MLP implementation
│   └── utils/                  # Utility functions
│       ├── __init__.py         # Utils initialization
│       ├── logging.py          # Logging utilities
│       ├── visualization.py    # Visualization utilities
│       └── export.py           # Export utilities
├── examples/                   # Example scripts
│   ├── __init__.py             # Examples initialization
│   ├── example.py              # Basic usage example
│   └── run_example.py          # Example runner
├── tests/                      # Test directory
│   ├── __init__.py             # Tests initialization
│   ├── test_metrics.py         # Tests correlation metrics
│   └── test_nas.py             # Tests zero-cost ensemble search
├── setup.py                    # Package setup script
├── requirements.txt            # Package dependencies
├── README.md                   # Project documentation
├── USAGE.md                    # Usage documentation
├── CONTRIBUTING.md             # Contribution guidelines
├── Makefile                    # Project Makefile
├── MANIFEST.in                 # Package manifest
├── test_local_install.py       # Test for local installation
└── LICENSE                     # License information
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/IgorSadoune/zcsearch.git
   cd zcs
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .[dev]
   ```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
- Use 4 spaces for indentation (no tabs).
- Maximum line length is 88 characters (compatible with Black).
- Use meaningful variable and function names.
- Write docstrings for all functions, classes, and modules following the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Type Annotations

- Use type annotations for function arguments and return values.
- Import types from the `typing` module.

Example:
```python
from typing import List, Dict, Optional

def process_data(data: List[float], config: Optional[Dict[str, any]] = None) -> float:
    """Process the input data.
    
    Args:
        data: List of data points to process.
        config: Optional configuration dictionary.
        
    Returns:
        Processed result as a float.
    """
    # Implementation
    return result
```

### Code Formatting

We use the following tools for code formatting and linting:

- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](http://mypy-lang.org/) for type checking

You can run these tools as follows:

```bash
# Format code with Black
black zcsearch tests examples

# Sort imports
isort zcsearch tests examples

# Lint with flake8
flake8 zcsearch tests examples

# Type check with mypy
mypy zcsearch
```

## Git Workflow

### Branching Strategy

- `main`: Main branch containing stable code.
- `develop`: Development branch for integrating features.
- `feature/<feature-name>`: Feature branches for new features.
- `bugfix/<bug-name>`: Bugfix branches for fixing bugs.

### Commit Messages

Use the following template for commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Where:
- `<type>`: Type of change (feat, fix, docs, style, refactor, test, chore)
- `<scope>`: Scope of the change (e.g., metrics, models, utils)
- `<subject>`: Short description of the change
- `<body>`: Detailed description of the change
- `<footer>`: References to issues, breaking changes, etc.

Example:
```
feat(metrics): add ZiCo metric implementation

Implement the ZiCo metric from Li et al. (2022) for zero-cost evaluation.

Closes #42
```

## Pull Request Process

1. Create a new branch from `develop` for your changes.
2. Make your changes and commit them with descriptive commit messages.
3. Push your branch to the repository.
4. Create a pull request to merge your branch into `develop`.
5. Ensure all tests pass and the code meets the coding standards.
6. Request a review from a maintainer.
7. Address any feedback from the review.
8. Once approved, your changes will be merged.

## Testing

We use [pytest](https://pytest.org/) for testing. Write tests for all new features and bug fixes.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=zero_cost_search
```

### Writing Tests

- Place tests in the `tests/` directory.
- Name test files with the prefix `test_`.
- Name test functions with the prefix `test_`.
- Use descriptive test function names that explain what is being tested.

Example:
```python
# tests/test_metrics.py
import torch
import pytest
from zero_cost_search import ZeroCostMetrics

def test_zico_score_positive_for_valid_model():
    # Setup
    model = create_test_model()
    X = torch.randn(10, 5)
    device = torch.device('cpu')
    
    # Execute
    score = ZeroCostMetrics.compute_zico_score(model, X, device)
    
    # Assert
    assert score > 0, "ZiCo score should be positive for a valid model"
```

Thank you for contributing to Zero-Cost Search!