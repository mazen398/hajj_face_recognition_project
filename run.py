"""Drop a file in data/, set the three lines below, then: python run.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pca import PCA

DATA_PATH = Path("data/iris.csv")  # your csv / tsv / xlsx
LABEL_COL = "species"  # color column, or None
N_COMPONENTS = 2


def load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


frame = load(DATA_PATH)
features = [
    c
    for c in frame.columns
    if pd.api.types.is_numeric_dtype(frame[c]) and c != LABEL_COL
]
keep = features + ([LABEL_COL] if LABEL_COL else [])
work = frame[keep].dropna()
X = work[features].to_numpy()

model = PCA(n_components=N_COMPONENTS)
scores = model.fit_transform(X)

print(f"{DATA_PATH.name}: {X.shape[0]} rows, {len(features)} features -> {N_COMPONENTS} PCs")
print("features:", ", ".join(features))
print("explained variance:")
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
score_frame = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
if LABEL_COL:
    score_frame[LABEL_COL] = work[LABEL_COL].to_numpy()
score_frame.to_csv(out / "scores.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 3.5))
xs = range(1, len(model.explained_variance_ratio_) + 1)
ax.bar(xs, model.explained_variance_ratio_)
ax.set_xlabel("PC")
ax.set_ylabel("explained variance")
fig.tight_layout()
fig.savefig(out / "scree.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
y = scores[:, 1] if scores.shape[1] > 1 else [0] * len(scores)
if LABEL_COL:
    for name, grp in score_frame.groupby(LABEL_COL):
        yy = grp["PC2"] if "PC2" in grp else [0] * len(grp)
        ax.scatter(grp["PC1"], yy, label=str(name), s=28)
    ax.legend(frameon=False)
else:
    ax.scatter(score_frame["PC1"], y, s=28)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(out / "scatter.png", dpi=140)
plt.close(fig)

print(f"\nwrote {out / 'scores.csv'}")
print(f"wrote {out / 'scree.png'}")
print(f"wrote {out / 'scatter.png'}")
