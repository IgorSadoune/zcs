#!/usr/bin/env python3

"""
Zero-Cost Neural Architecture Search Example

This example demonstrates how to use the zero_cost_search package
to find a good neural network architecture without training.
"""

import os
import sys
import time

import numpy as np
import torch

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zcsearch import ZeroCostNAS, plot_search_results, setup_logger


def main():
    # Set up logging
    logger = setup_logger(level="INFO")
    logger.info("Starting Zero-Cost NAS example")

    # Create synthetic data for demonstration
    input_dim = 20
    output_dim = 3  # Classification with 3 classes
    num_samples = 200

    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))

    logger.info(
        f"Created synthetic dataset with {num_samples} samples, "
        f"{input_dim} features, and {output_dim} classes"
    )

    # Create cache directory
    cache_dir = "./nas_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize the NAS
    nas = ZeroCostNAS(
        input_dim=input_dim, output_dim=output_dim, seed=42, cache_dir=cache_dir
    )

    # Option 1: Quick prediction using meta-learning only
    print("\n=== Option 1: Quick Architecture Prediction ===\n")
    start_time = time.time()
    predicted_config = nas.predict_architecture(X, y)
    prediction_time = time.time() - start_time

    print(
        f"Predicted architecture using meta-learning (took {prediction_time:.2f} seconds):"
    )
    print(f"  Hidden dimensions: {predicted_config['hidden_dims']}")
    print(f"  Activation function: {predicted_config['activation_fn_str']}")

    # Option 2: Full search with zero-cost proxies
    print("\n=== Option 2: Full Architecture Search ===\n")
    result = nas.search(
        X=X,
        y=y,
        depths=[2, 3, 4],
        widths=[64, 128, 256],
        activations=["relu", "tanh", "leaky_relu"],
        num_samples=15,
        use_meta_learning=True,  # Use meta-learning to guide the search
        verbose=True,
    )

    # Get the best model (optionally with dropout and batchnorm for training)
    best_model = nas.get_best_model(include_dropout=0.2, include_batchnorm=True)
    print("\nBest model architecture:")
    print(best_model)

    # Plot the results
    plot_search_results(result, save_path="search_results.png")

    # Get detailed results summary
    summary = nas.get_results_summary()

    print("\n=== Architecture Search Summary ===\n")
    print(f"Search time: {result['search_time']:.2f} seconds")
    print(f"Best configuration: {summary['best_config']}")
    print(f"Best score: {summary['best_score']:.4f}")

    print("\nTop 5 configurations:")
    for i, config in enumerate(summary["top_configurations"]):
        print(f"  {i+1}. {config['config']} (score: {config['ensemble_score']:.4f})")

    print("\nArchitectural choice analysis:")
    print("  Network depth impact:")
    for depth, score in summary["depth_analysis"].items():
        print(f"    Depth {depth}: {score:.4f}")

    print("  Activation function impact:")
    for act, score in summary["activation_analysis"].items():
        print(f"    {act}: {score:.4f}")


if __name__ == "__main__":
    main()
