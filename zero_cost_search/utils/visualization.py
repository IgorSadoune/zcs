import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_search_results(
    results: Dict[str, Any], save_path: Optional[str] = None, show_plot: bool = True
) -> None:
    """Plot the results of the architecture search.

    Args:
        results: Dictionary containing search results from ZeroCostNAS.search()
        save_path: Path to save the plot. If None, the plot is not saved.
        show_plot: Whether to display the plot.
    """
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        )
        return

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Zero-Cost NAS Search Results", fontsize=16)

    # Plot 1: Top configurations by score
    if "all_results" in results:
        # Sort configurations by score
        sorted_results = sorted(
            results["all_results"], key=lambda x: x["scores"]["ensemble"], reverse=True
        )

        # Get top 10 or all if less than 10
        top_n = min(10, len(sorted_results))
        top_configs = sorted_results[:top_n]

        # Extract scores and labels
        scores = [result["scores"]["ensemble"] for result in top_configs]
        labels = []
        for result in top_configs:
            config = result["config"]
            label = f"{config['activation_fn_str']}, {config['hidden_dims']}"
            labels.append(label)

        # Plot horizontal bar chart
        y_pos = np.arange(len(scores))
        axs[0, 0].barh(y_pos, scores, align="center")
        axs[0, 0].set_yticks(y_pos)
        axs[0, 0].set_yticklabels(labels)
        axs[0, 0].invert_yaxis()  # Labels read top-to-bottom
        axs[0, 0].set_xlabel("Ensemble Score")
        axs[0, 0].set_title("Top Configurations")
        axs[0, 0].grid(axis="x", linestyle="--", alpha=0.7)

    # Plot 2: Score distribution
    if "all_results" in results:
        all_scores = [result["scores"]["ensemble"] for result in results["all_results"]]
        axs[0, 1].hist(
            all_scores, bins=10, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axs[0, 1].axvline(
            results["best_score"],
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f'Best score: {results["best_score"]:.4f}',
        )
        axs[0, 1].set_xlabel("Ensemble Score")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].set_title("Score Distribution")
        axs[0, 1].legend()
        axs[0, 1].grid(linestyle="--", alpha=0.7)

    # Plot 3: Metric breakdown for best configuration
    if "best_scores" in results:
        # Filter out the ensemble score
        metric_scores = {
            k: v for k, v in results["best_scores"].items() if k != "ensemble"
        }
        metrics = list(metric_scores.keys())
        scores = list(metric_scores.values())

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        # Close the polygon
        angles += [angles[0]]
        scores += [scores[0]]

        axs[1, 0].plot(angles, scores, "o-", linewidth=2)
        axs[1, 0].fill(angles, scores, alpha=0.25)
        axs[1, 0].set_xticks(angles[:-1])
        axs[1, 0].set_xticklabels(metrics)
        # Set radial limits for polar plots
        axs[1, 0].set_ylim(0, 1)  # For older matplotlib versions, use set_ylim for radial axis
        axs[1, 0].set_title("Metric Breakdown for Best Configuration")
        axs[1, 0].grid(True)

    # Plot 4: Analysis by architectural choice
    if "all_results" in results:
        # Analyze impact of different architectural choices
        depth_scores = {}
        activation_scores = {}

        for result in results["all_results"]:
            config = result["config"]
            score = result["scores"]["ensemble"]

            # By depth
            depth = len(config["hidden_dims"])
            if depth not in depth_scores:
                depth_scores[depth] = []
            depth_scores[depth].append(score)

            # By activation
            act = config["activation_fn_str"]
            if act not in activation_scores:
                activation_scores[act] = []
            activation_scores[act].append(score)

        # Compute average score for each choice
        depth_avg = {d: np.mean(scores) for d, scores in depth_scores.items()}
        act_avg = {a: np.mean(scores) for a, scores in activation_scores.items()}

        # Plot as grouped bar chart
        x = np.arange(2)  # Two groups: depth and activation
        width = 0.8 / max(len(depth_avg), len(act_avg))  # Width of bars

        # Plot depths
        for i, (depth, score) in enumerate(sorted(depth_avg.items())):
            axs[1, 1].bar(
                x[0] + (i - len(depth_avg) / 2 + 0.5) * width,
                score,
                width,
                label=f"Depth {depth}",
            )

        # Plot activations
        for i, (act, score) in enumerate(sorted(act_avg.items())):
            axs[1, 1].bar(
                x[1] + (i - len(act_avg) / 2 + 0.5) * width,
                score,
                width,
                label=f"{act}",
            )

        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(["Network Depth", "Activation Function"])
        axs[1, 1].set_ylabel("Average Score")
        axs[1, 1].set_title("Impact of Architectural Choices")
        axs[1, 1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot if requested
    if save_path is not None:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
