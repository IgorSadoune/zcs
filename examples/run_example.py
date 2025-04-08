#!/usr/bin/env python3

"""
Simple example script for Zero-Cost Neural Architecture Search

This script can be run directly to demonstrate the basic functionality
of the zcsearch package.

Usage:
    python run_example.py
"""

import os
import sys

import torch

# Try to import directly first (if package is installed)
try:
    from zcsearch import ZeroCostNAS
except ImportError:
    # If that fails, add the parent directory to the path (for development)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from zcsearch import ZeroCostNAS


def main():
    print("\n=== Zero-Cost Neural Architecture Search Example ===\n")

    # Create synthetic data
    input_dim = 10
    output_dim = 2  # Binary classification
    num_samples = 100

    print(
        f"Creating synthetic dataset with {num_samples} samples, "
        f"{input_dim} features, and {output_dim} classes..."
    )

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    # Initialize the NAS
    print("\nInitializing Zero-Cost NAS...")
    nas = ZeroCostNAS(input_dim=input_dim, output_dim=output_dim, seed=42)

    # Predict architecture using meta-learning
    print("\nPredicting architecture using meta-learning...")
    predicted_config = nas.predict_architecture(X, y)

    print(f"\nPredicted architecture:")
    print(f"  Hidden dimensions: {predicted_config['hidden_dims']}")
    print(f"  Activation function: {predicted_config['activation_fn_str']}")

    # Run a small search
    print("\nRunning a small architecture search...")
    result = nas.search(
        X=X,
        y=y,
        depths=[2, 3],
        widths=[32, 64],
        activations=["relu", "tanh"],
        num_samples=4,
        use_meta_learning=True,
        verbose=True,
    )

    # Get the best model
    best_model = nas.get_best_model()

    print("\nSearch completed!")
    print(f"Best architecture: {result['best_config']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Search time: {result['search_time']:.2f} seconds")

    # Make a prediction with the best model
    print("\nMaking a prediction with the best model...")
    with torch.no_grad():
        sample_input = torch.randn(1, input_dim)
        output = best_model(sample_input)
        predicted_class = output.argmax(dim=1).item()

    print(f"Predicted class: {predicted_class}")
    print("\nDone!")


if __name__ == "__main__":
    main()
