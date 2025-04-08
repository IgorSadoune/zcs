#!/usr/bin/env python3

"""
Command-line interface for Zero-Cost Neural Architecture Search

This module provides a command-line interface for running
zero-cost neural architecture search on datasets.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .utils import plot_search_results, setup_logger
from .zero_cost_search import ZeroCostNAS


def load_data(data_path: str) -> tuple:
    """Load data from a file.

    Supports .pt (PyTorch), .npy (NumPy), and .csv files.

    Args:
        data_path: Path to the data file.

    Returns:
        Tuple of (X, y) tensors.
    """
    logger = logging.getLogger("zero_cost_search")
    logger.info(f"Loading data from {data_path}")

    if data_path.endswith(".pt"):
        # Load PyTorch tensor
        data = torch.load(data_path)
        if isinstance(data, tuple) and len(data) == 2:
            X, y = data
        elif isinstance(data, dict) and "X" in data and "y" in data:
            X, y = data["X"], data["y"]
        else:
            raise ValueError(
                f"Unsupported data format in {data_path}. Expected tuple (X, y) or dict with 'X' and 'y' keys."
            )

    elif data_path.endswith(".npy"):
        # Load NumPy array
        data = np.load(data_path, allow_pickle=True)
        if (
            isinstance(data, np.ndarray)
            and data.dtype == np.dtype("O")
            and len(data) == 2
        ):
            X = torch.tensor(data[0], dtype=torch.float32)
            y = torch.tensor(data[1], dtype=torch.long)
        else:
            raise ValueError(
                f"Unsupported data format in {data_path}. Expected array with X and y."
            )

    elif data_path.endswith(".csv"):
        # Load CSV file
        try:
            import pandas as pd

            df = pd.read_csv(data_path)

            # Assume last column is target
            X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
            y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        except ImportError:
            logger.error(
                "Pandas is required for loading CSV files. Install with 'pip install pandas'."
            )
            sys.exit(1)
    else:
        raise ValueError(
            f"Unsupported file format: {data_path}. Supported formats: .pt, .npy, .csv"
        )

    logger.info(
        f"Loaded data with {X.shape[0]} samples, {X.shape[1]} features, and {len(torch.unique(y))} classes"
    )
    return X, y


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save search results to a file.

    Args:
        results: Search results dictionary.
        output_path: Path to save the results.
    """
    logger = logging.getLogger("zero_cost_search")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert torch tensors to lists for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if k == "all_results":
            # Skip detailed results to keep the file smaller
            continue
        elif k == "best_model_state":
            # Skip model state
            continue
        elif isinstance(v, torch.Tensor):
            serializable_results[k] = v.tolist()
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, dict):
            # Handle nested dictionaries
            serializable_results[k] = {}
            for kk, vv in v.items():
                if isinstance(vv, (torch.Tensor, np.ndarray)):
                    serializable_results[k][kk] = vv.tolist()
                else:
                    serializable_results[k][kk] = vv
        else:
            serializable_results[k] = v

    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Zero-Cost Neural Architecture Search")

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data file (.pt, .npy, or .csv)",
    )

    # Model arguments
    parser.add_argument(
        "--input-dim",
        type=int,
        help="Input dimension (inferred from data if not provided)",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        help="Output dimension (inferred from data if not provided)",
    )

    # Search arguments
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Network depths to consider (default: [2, 3, 4])",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Layer widths to consider (default: [64, 128, 256])",
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=["relu", "tanh", "leaky_relu"],
        help='Activation functions to consider (default: ["relu", "tanh", "leaky_relu"])',
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of architectures to evaluate (default: 20)",
    )
    parser.add_argument(
        "--no-meta-learning",
        action="store_true",
        help="Disable meta-learning for architecture prediction",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to save the results (default: results.json)",
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Path to save the plot (default: None)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache results (default: None)",
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to save the log (default: None)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(level=args.log_level, log_file=args.log_file)
    logger.info("Starting Zero-Cost Neural Architecture Search")

    # Load data
    X, y = load_data(args.data)

    # Infer dimensions if not provided
    input_dim = args.input_dim if args.input_dim is not None else X.shape[1]
    output_dim = (
        args.output_dim if args.output_dim is not None else len(torch.unique(y))
    )

    # Initialize the NAS
    nas = ZeroCostNAS(
        input_dim=input_dim,
        output_dim=output_dim,
        seed=args.seed,
        cache_dir=args.cache_dir,
        log_level=args.log_level,
    )

    # Run the search
    result = nas.search(
        X=X,
        y=y,
        depths=args.depths,
        widths=args.widths,
        activations=args.activations,
        num_samples=args.num_samples,
        use_meta_learning=not args.no_meta_learning,
        verbose=True,
    )

    # Get the best model
    best_model = nas.get_best_model()

    # Print results
    print("\nSearch completed!")
    print(f"Best architecture: {result['best_config']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Search time: {result['search_time']:.2f} seconds")

    # Save results
    save_results(result, args.output)

    # Plot results if requested
    if args.plot is not None:
        plot_search_results(result, save_path=args.plot, show_plot=False)
        print(f"Plot saved to {args.plot}")

    print("\nDone!")


if __name__ == "__main__":
    main()