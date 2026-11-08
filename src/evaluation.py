"""Save the CSV, comparison graph and confusion matrices for one dataset."""

# Standard libraries
from pathlib import Path

# Plotting
import matplotlib

# Use a non-interactive backend, must be set before pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Machine learning libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Above this many identities, axis labels are left off
MAX_LABELS_ON_AXIS = 25


def save_comparison_csv(results, output_dir):
    """Save final_model_comparison.csv."""
    table = pd.DataFrame(
        {
            "Model": [r["name"] for r in results],
            "Test Accuracy": [round(r["accuracy"], 4) for r in results],
        }
    )
    output_path = Path(output_dir) / "final_model_comparison.csv"
    table.to_csv(output_path, index=False)
    return output_path


def save_accuracy_comparison_chart(results, dataset_name, output_dir):
    """Create the accuracy comparison graph."""
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    # One colour per model
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accuracies, color=colors, width=0.55)

    # Label each bar with its accuracy
    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.02,
            f"{accuracy:.2%}",
            ha="center",
            fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy comparison - {dataset_name}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Save graph
    fig.tight_layout()
    output_path = Path(output_dir) / "accuracy_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_confusion_matrix(y_true, y_predicted, title, output_path):
    """Create a confusion matrix: rows are true identities, columns predicted."""
    # Fixed label order so every matrix matches
    labels = np.unique(np.concatenate([y_true, y_predicted]))
    matrix = confusion_matrix(y_true, y_predicted, labels=labels)

    size = max(6, min(18, 0.45 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.8, label="number of test images")

    ax.set_xlabel("Predicted identity")
    ax.set_ylabel("True identity")
    ax.set_title(title)

    # With few identities, show names and counts
    if len(labels) <= MAX_LABELS_ON_AXIS:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        threshold = matrix.max() / 2 if matrix.max() else 0
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                if value:
                    ax.text(
                        col,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > threshold else "black",
                    )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Predicted identity ({len(labels)} identities)")
        ax.set_ylabel(f"True identity ({len(labels)} identities)")

    # Save figure
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
