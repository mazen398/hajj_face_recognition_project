"""Matplotlib figures used by the Streamlit tester and the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pca.pca import PCA


def scree_figure(model: PCA) -> Figure:
    ratios = model.explained_variance_ratio_
    cumulative = model.cumulative_variance_ratio()
    xs = np.arange(1, len(ratios) + 1)

    fig = Figure(figsize=(8.2, 4.2), layout="constrained")
    ax = fig.add_subplot(111)
    ax.bar(xs, ratios, color="#3b6fd6", width=0.62, label="share of variance")
    ax.plot(xs, cumulative, color="#d35400", marker="o", linewidth=2, label="cumulative")
    ax.set_xticks(xs)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def scatter_figure(scores: np.ndarray, labels: pd.Series | None = None) -> Figure:
    fig = Figure(figsize=(7.4, 5.6), layout="constrained")
    ax = fig.add_subplot(111)
    x = scores[:, 0]
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))

    if labels is None:
        ax.scatter(x, y, s=36, c="#3b6fd6", alpha=0.85, edgecolors="white", linewidths=0.4)
    else:
        groups = pd.Series(labels).astype(str)
        palette = [
            "#3b6fd6",
            "#d35400",
            "#1e8449",
            "#8e44ad",
            "#16a085",
            "#c0392b",
            "#2c3e50",
            "#f1c40f",
        ]
        for i, name in enumerate(groups.unique()):
            mask = groups == name
            ax.scatter(
                x[mask],
                y[mask],
                s=36,
                c=palette[i % len(palette)],
                alpha=0.88,
                edgecolors="white",
                linewidths=0.4,
                label=name,
            )
        ax.legend(frameon=False, title="label")

    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2" if scores.shape[1] > 1 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def loadings_frame(model: PCA, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        model.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(model.n_components_)],
    )
