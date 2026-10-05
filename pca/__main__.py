"""Run PCA from the command line.

    python -m pca data/iris.csv --label species --components 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca.io import load_table, numeric_columns
from pca.pca import PCA


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA on a table.")
    parser.add_argument("path", help="CSV / TSV / Excel file")
    parser.add_argument("--label", default=None, help="Optional color/group column")
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--out", default=None, help="Write scores CSV here")
    args = parser.parse_args()

    frame = load_table(args.path)
    features = [c for c in numeric_columns(frame) if c != args.label]
    if not features:
        raise SystemExit("No numeric columns to use.")

    X = frame[features].dropna()
    model = PCA(n_components=args.components, standardize=not args.no_standardize)
    scores = model.fit_transform(X.to_numpy())

    print(f"rows={X.shape[0]}  features={len(features)}  components={model.n_components_}")
    print("explained variance ratio:")
    for i, (ratio, cumulative) in enumerate(
        zip(model.explained_variance_ratio_, model.cumulative_variance_ratio()), start=1
    ):
        print(f"  PC{i}: {ratio:.4f}   cumulative {cumulative:.4f}")

    score_frame = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    if args.label and args.label in frame.columns:
        score_frame[args.label] = frame.loc[X.index, args.label]

    out = Path(args.out) if args.out else Path(args.path).with_name(Path(args.path).stem + "_scores.csv")
    score_frame.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
