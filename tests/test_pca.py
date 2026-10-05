import numpy as np
import pandas as pd
import pytest

from pca.io import load_table, numeric_columns
from pca.pca import PCA


def test_iris_two_components_cover_most_variance():
    frame = load_table("data/iris.csv")
    features = [c for c in numeric_columns(frame) if c != "species"]
    X = frame[features].to_numpy()
    model = PCA(n_components=2).fit(X)
    assert model.cumulative_variance_ratio()[-1] > 0.9


def test_scores_have_requested_width():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    scores = PCA(n_components=3).fit_transform(X)
    assert scores.shape == (40, 3)


def test_reconstruction_improves_with_more_components():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    err1 = _recon_error(X, 1)
    err4 = _recon_error(X, 4)
    assert err4 < err1
    assert err4 < 1e-10


def test_rejects_nan():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="NaN"):
        PCA().fit(X)


def test_rejects_one_row():
    with pytest.raises(ValueError, match="2 rows"):
        PCA().fit(np.array([[1.0, 2.0]]))


def test_column_mismatch():
    rng = np.random.default_rng(2)
    model = PCA(n_components=1).fit(rng.normal(size=(10, 3)))
    with pytest.raises(ValueError, match="Expected 3 columns"):
        model.transform(rng.normal(size=(4, 2)))


def test_numeric_columns_skip_text():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [0.1, 0.2]})
    assert numeric_columns(frame) == ["a", "c"]


def _recon_error(X: np.ndarray, k: int) -> float:
    model = PCA(n_components=k, standardize=False).fit(X)
    rebuilt = model.inverse_transform(model.transform(X))
    return float(np.mean((X - rebuilt) ** 2))
