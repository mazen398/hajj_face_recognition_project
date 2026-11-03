"""
EVALUATION
==========

WHAT COMES IN : the true test labels, the predicted test labels, and the
                accuracy of each experiment.
WHAT HAPPENS  : the numbers are turned into the files that go in the report.
WHAT GOES OUT : a CSV table, an accuracy bar chart, and one confusion matrix
                image per experiment.

WHY THIS STEP IS NECESSARY
--------------------------
A single accuracy number says how often the system was right, but not WHERE it
went wrong. The confusion matrix shows which identity was mistaken for which,
which is the interesting part when you have to explain the result.

Every file is written inside the folder belonging to its own dataset, so
dataset1 results and dataset2 results are never written to the same place.
"""

from pathlib import Path

import matplotlib

# "Agg" draws charts straight to image files without needing a screen.
# This must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Above this many identities the axis labels become an unreadable smear,
# so we draw the matrix without text labels instead.
MAX_LABELS_ON_AXIS = 25


def save_comparison_csv(results, output_dir):
    """Write final_model_comparison.csv with one row per experiment."""
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
    """Bar chart comparing the final test accuracy of the two experiments."""
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accuracies, color=["#4C72B0", "#DD8452"], width=0.55)

    # Print the exact value on top of each bar so the chart is readable alone.
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

    fig.tight_layout()
    output_path = Path(output_dir) / "accuracy_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_confusion_matrix(y_true, y_predicted, title, output_path):
    """Draw and save one confusion matrix.

    Rows    = the identity the photo actually belongs to.
    Columns = the identity the system predicted.
    A perfect system puts every count on the diagonal.
    """
    # labels= fixes the row/column order so it is identical in both matrices.
    labels = np.unique(np.concatenate([y_true, y_predicted]))
    matrix = confusion_matrix(y_true, y_predicted, labels=labels)

    size = max(6, min(18, 0.45 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.8, label="number of test images")

    ax.set_xlabel("Predicted identity")
    ax.set_ylabel("True identity")
    ax.set_title(title)

    if len(labels) <= MAX_LABELS_ON_AXIS:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # With few identities there is room to print the counts in the cells.
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
