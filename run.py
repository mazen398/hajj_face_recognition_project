"""
1. Put your table in the data/ folder.
2. Set TRAIN_PATH to that file.
3. If you also have a test table, set TEST_PATH. Otherwise leave it None.
4. Run: python run.py
5. Read the printout and the files in outputs/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pca import PCA

# --- change these ---
TRAIN_PATH = Path("data/iris.csv")  # your training csv / tsv / xlsx
TEST_PATH = None  # example: Path("data/test.csv")
LABEL_COL = "species"  # name of the group column, or None
N_COMPONENTS = 2
# --------------------


def load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def features_of(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and c != LABEL_COL
    ]


def matrix(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, object]:
    keep = features + ([LABEL_COL] if LABEL_COL else [])
    work = frame[keep].dropna()
    return work, work[features].to_numpy()


def write_scores(path: Path, scores, work: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    if LABEL_COL:
        out[LABEL_COL] = work[LABEL_COL].to_numpy()
    out.to_csv(path, index=False)
    return out


train = load(TRAIN_PATH)
features = features_of(train)
train_work, X_train = matrix(train, features)

model = PCA(n_components=N_COMPONENTS)
train_scores = model.fit_transform(X_train)

print(f"train  {TRAIN_PATH.name}: {X_train.shape[0]} rows, {len(features)} features")
print("features:", ", ".join(features))
print("explained variance (fit on train):")
for i, ratio in enumerate(model.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio:.1%}")

loadings = pd.DataFrame(
    model.components_.T,
    index=features,
    columns=[f"PC{i+1}" for i in range(model.n_components_)],
)
print("\nloadings:")
print(loadings.round(3).to_string())

out = Path("outputs")
out.mkdir(exist_ok=True)
train_frame = write_scores(out / "train_scores.csv", train_scores, train_work)
print(f"\nwrote {out / 'train_scores.csv'}")

if TEST_PATH is not None:
    test = load(TEST_PATH)
    missing = [c for c in features if c not in test.columns]
    if missing:
        raise SystemExit(f"Test file is missing columns: {missing}")
    test_work, X_test = matrix(test, features)
    test_scores = model.transform(X_test)
    write_scores(out / "test_scores.csv", test_scores, test_work)
    print(f"test   {TEST_PATH.name}: {X_test.shape[0]} rows")
    print(f"wrote {out / 'test_scores.csv'}")

fig, ax = plt.subplots(figsize=(6, 3.5))
xs = range(1, len(model.explained_variance_ratio_) + 1)
ax.bar(list(xs), model.explained_variance_ratio_)
ax.set_xlabel("PC")
ax.set_ylabel("explained variance")
fig.tight_layout()
fig.savefig(out / "scree.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
y = train_scores[:, 1] if train_scores.shape[1] > 1 else [0] * len(train_scores)
if LABEL_COL:
    for name, grp in train_frame.groupby(LABEL_COL):
        yy = grp["PC2"] if "PC2" in grp else [0] * len(grp)
        ax.scatter(grp["PC1"], yy, label=str(name), s=28)
    ax.legend(frameon=False)
else:
    ax.scatter(train_frame["PC1"], y, s=28)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("train scores")
fig.tight_layout()
fig.savefig(out / "scatter.png", dpi=140)
plt.close(fig)

print(f"wrote {out / 'scree.png'}")
print(f"wrote {out / 'scatter.png'}")
