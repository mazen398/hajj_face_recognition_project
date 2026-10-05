"""
PCA on photos. One file. Nothing else needed.

HOW TO USE
1. Put the folder path of YOUR photos in PHOTOS_FOLDER below.
   You do NOT need to move or copy your photos anywhere.
2. Run:  python pca_photos.py
3. Open the "pca_results" folder it creates next to this file.

To EDIT this file, right-click it and choose Open with > Notepad.
Double-clicking it RUNS it instead of opening it.
"""

import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============== CHANGE THIS ONE LINE ==============
PHOTOS_FOLDER = r"C:\Users\darks\Desktop\my_photos"
# ==================================================

IMAGE_SIZE = 64  # each photo is resized to 64x64
N_COMPONENTS = 50  # how many patterns to learn
TEST_SHARE = 0.2  # last 20% of photos are held out as the test set
MAX_PHOTOS = 4000  # safety cap so thousands of photos still fit in memory

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Z = X - self.mean_
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        var = (s**2) / (X.shape[0] - 1)
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores):
        return scores @ self.components_ + self.mean_


def load_photos(folder):
    if not folder.exists():
        raise SystemExit(
            f"\nThat folder does not exist:\n  {folder}\n\n"
            "Open this file in Notepad and fix the PHOTOS_FOLDER line near the top."
        )
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(
            f"\nNo photos found in:\n  {folder}\n\n"
            "Fix the PHOTOS_FOLDER line at the top of this file so it points\n"
            "at the folder where your pictures actually are."
        )
    if len(paths) > MAX_PHOTOS:
        print(f"Found {len(paths)} photos. Using the first {MAX_PHOTOS}.")
        paths = paths[:MAX_PHOTOS]

    rows, kept = [], []
    for i, path in enumerate(paths, start=1):
        try:
            img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        except Exception:
            continue
        rows.append(np.asarray(img, dtype=float).reshape(-1) / 255.0)
        kept.append(path)
        if i % 250 == 0:
            print(f"  loaded {i}/{len(paths)}")
    if len(rows) < 4:
        raise SystemExit("Need at least 4 readable photos.")
    return np.stack(rows), kept


def grid(images, title, path, n=16):
    n = min(n, len(images))
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(8, 2 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def before_after(X, rebuilt, paths, path, n=6):
    n = min(n, len(X))
    fig, axes = plt.subplots(n, 2, figsize=(4.5, 2 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[i, 0].imshow(X[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(rebuilt[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(paths[i].name[:22], fontsize=7)
        axes[i, 1].set_title("rebuilt by PCA", fontsize=7)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    folder = Path(PHOTOS_FOLDER).expanduser()
    print(f"Looking for photos in: {folder}")
    X, paths = load_photos(folder)

    split = max(2, int(len(X) * (1 - TEST_SHARE)))
    X_train, X_test = X[:split], X[split:]
    train_paths, test_paths = paths[:split], paths[split:]
    print(f"\n{len(X)} photos total -> {len(X_train)} train, {len(X_test)} test")

    model = PCA(min(N_COMPONENTS, len(X_train) - 1)).fit(X_train)
    train_scores = model.transform(X_train)

    print("\nHow much each pattern explains:")
    total = 0.0
    for i, r in enumerate(model.explained_variance_ratio_[:10], start=1):
        total += r
        print(f"  pattern {i}: {r:6.1%}   running total {total:.1%}")
    print(f"  all {len(model.explained_variance_ratio_)} patterns keep "
          f"{model.explained_variance_ratio_.sum():.1%} of the detail")

    out = Path(__file__).parent / "pca_results"
    out.mkdir(exist_ok=True)

    header = "file," + ",".join(f"PC{i+1}" for i in range(train_scores.shape[1]))
    with open(out / "train_scores.csv", "w") as f:
        f.write(header + "\n")
        for p, row in zip(train_paths, train_scores):
            f.write(p.name + "," + ",".join(f"{v:.5f}" for v in row) + "\n")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(1, len(model.explained_variance_ratio_) + 1), model.explained_variance_ratio_)
    ax.set_xlabel("pattern number")
    ax.set_ylabel("share of detail")
    fig.tight_layout()
    fig.savefig(out / "how_much_each_pattern_matters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(model.mean_.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    ax.set_title("average photo")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "average_photo.png", dpi=140)
    plt.close(fig)

    patterns = model.components_.copy()
    for i, v in enumerate(patterns):
        patterns[i] = (v - v.min()) / (v.max() - v.min() + 1e-8)
    grid(patterns, "patterns PCA learned", out / "patterns_learned.png")

    before_after(X_train, model.inverse_transform(train_scores), train_paths,
                 out / "train_before_after.png")

    if len(X_test):
        test_scores = model.transform(X_test)
        rebuilt = model.inverse_transform(test_scores)
        error = np.mean((X_test - rebuilt) ** 2, axis=1)
        with open(out / "test_scores.csv", "w") as f:
            f.write(header + ",rebuild_error\n")
            for p, row, e in zip(test_paths, test_scores, error):
                f.write(p.name + "," + ",".join(f"{v:.5f}" for v in row) + f",{e:.6f}\n")
        before_after(X_test, rebuilt, test_paths, out / "test_before_after.png")
        print(f"\nTest photos average rebuild error: {error.mean():.5f}  (lower is better)")

    print(f"\nDONE. Open this folder to see everything:\n  {out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:
        print(stop)
    except Exception:
        traceback.print_exc()
    # Keeps the black window open when the file is launched by double-click.
    input("\nPress Enter to close this window...")
