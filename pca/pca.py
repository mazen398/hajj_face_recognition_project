"""Principal Component Analysis via SVD.

Center the matrix, optionally divide each column by its standard deviation,
then take the top-k right singular vectors. That is the whole method.
"""

from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, n_components: int | None = None, standardize: bool = True):
        if n_components is not None and int(n_components) < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.standardize = standardize
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.n_samples_: int | None = None
        self.n_features_: int | None = None
        self.n_components_: int | None = None

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 rows to fit PCA")
        if X.shape[1] < 1:
            raise ValueError("Need at least 1 numeric column")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values. Clean the table first.")

        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        if self.standardize:
            scale = centered.std(axis=0, ddof=0)
            scale[scale == 0] = 1.0
            self.scale_ = scale
            centered = centered / scale
        else:
            self.scale_ = np.ones(self.n_features_, dtype=float)

        # X = U S V^T. Rows of V^T are principal axes.
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        max_k = vt.shape[0]
        k = max_k if self.n_components is None else min(int(self.n_components), max_k)
        self.n_components_ = k

        variance = (singular_values**2) / (self.n_samples_ - 1)
        total = variance.sum()
        if total <= 0:
            raise ValueError("All columns have zero variance after centering.")

        self.components_ = vt[:k]
        self.singular_values_ = singular_values[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / total
        return self

    def transform(self, X) -> np.ndarray:
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} columns, got {X.shape[1]}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        centered = (X - self.mean_) / self.scale_
        return centered @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, scores) -> np.ndarray:
        self._require_fit()
        scores = np.asarray(scores, dtype=float)
        recovered = scores @ self.components_
        return recovered * self.scale_ + self.mean_

    def cumulative_variance_ratio(self) -> np.ndarray:
        self._require_fit()
        return np.cumsum(self.explained_variance_ratio_)

    def _require_fit(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before using this method.")
