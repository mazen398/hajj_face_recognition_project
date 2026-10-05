"""PCA by SVD. Center columns, optionally scale them, take the top axes."""

from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, n_components: int | None = None, standardize: bool = True):
        self.n_components = n_components
        self.standardize = standardize

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("Need a 2D table with at least 2 rows.")
        if not np.isfinite(X).all():
            raise ValueError("Clean NaN / inf values first.")

        self.mean_ = X.mean(axis=0)
        Z = X - self.mean_
        if self.standardize:
            self.scale_ = Z.std(axis=0, ddof=0)
            self.scale_[self.scale_ == 0] = 1.0
            Z = Z / self.scale_
        else:
            self.scale_ = np.ones(X.shape[1])

        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        k = vt.shape[0] if self.n_components is None else min(int(self.n_components), vt.shape[0])
        var = (s**2) / (X.shape[0] - 1)
        self.n_components_ = k
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Z @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
