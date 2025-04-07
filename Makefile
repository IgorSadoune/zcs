# Makefile for zero_cost_search package

.PHONY: install test clean lint format example

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

test:
	pytest

test-cov:
	pytest --cov=zero_cost_search

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Linting and formatting
lint:
	flake8 zero_cost_search tests examples

format:
	black zero_cost_search tests examples
	isort zero_cost_search tests examples

# Run examples
example:
	python examples/run_example.py

full-example:
	python examples/example.py

# Build package
build:
	python setup.py sdist bdist_wheel

# Check package
check-package:
    twine check dist/*

# Upload to TestPyPI
upload-test:
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
upload:
    twine upload dist/*
	
# Or test first on TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test local installation
test-install: clean install
	python test_local_install.py