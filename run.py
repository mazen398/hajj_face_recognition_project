"""
PCA on photos.

1. Put training photos in data/train
2. Put test photos in data/test  (optional)
3. python run.py
4. Open outputs/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pca import PCA

# --- change these if you need to ---
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
IMAGE_SIZE = 64  # every photo is resized to 64x64
N_COMPONENTS = 50
# -----------------------------------

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
    )


def load_one(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=float).reshape(-1) / 255.0


def load_folder(folder: Path) -> tuple[np.ndarray, list[Path], list[str]]:
    paths = list_images(folder)
    if len(paths) < 2:
        raise SystemExit(f"Need at least 2 images in {folder}. Put your photos there.")
    rows = []
    labels = []
    for i, path in enumerate(paths, start=1):
        rows.append(load_one(path))
        parent = path.parent
        labels.append(parent.name if parent != folder else "")
        if i % 200 == 0:
            print(f"  loaded {i}/{len(paths)} from {folder}")
    return np.stack(rows), paths, labels


def show_as_images(vectors, title: str, path: Path, n: int = 16) -> None:
    n = min(n, len(vectors))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(vectors[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def reconstructions(X, scores, model, paths, out_path: Path, n: int = 8) -> None:
    n = min(n, len(X))
    rebuilt = model.inverse_transform(scores[:n])
    fig, axes = plt.subplots(n, 2, figsize=(4.5, 2 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        axes[i, 0].imshow(X[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(rebuilt[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(paths[i].name, fontsize=8)
        axes[i, 1].set_title("reconstructed", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


print(f"loading train from {TRAIN_DIR}")
X_train, train_paths, train_labels = load_folder(TRAIN_DIR)
# Images: center only. Do not scale each pixel like a spreadsheet column.
model = PCA(n_components=min(N_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1]), standardize=False)
train_scores = model.fit_transform(X_train)

print(f"train: {len(train_paths)} photos, {IMAGE_SIZE}x{IMAGE_SIZE} grayscale")
print("explained variance (fit on train):")
cumulative = 0.0
for i, ratio in enumerate(model.explained_variance_ratio_, start=1):
    cumulative += ratio
    if i <= 10 or i == len(model.explained_variance_ratio_):
        print(f"  PC{i}: {ratio:.1%}   cumulative {cumulative:.1%}")

out = Path("outputs")
out.mkdir(exist_ok=True)
score_cols = [f"PC{i+1}" for i in range(train_scores.shape[1])]
train_frame = pd.DataFrame(train_scores, columns=score_cols)
train_frame.insert(0, "file", [p.name for p in train_paths])
if any(train_labels):
    train_frame.insert(1, "label", train_labels)
train_frame.to_csv(out / "train_scores.csv", index=False)
print(f"wrote {out / 'train_scores.csv'}")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(range(1, len(model.explained_variance_ratio_) + 1), model.explained_variance_ratio_)
ax.set_xlabel("PC")
ax.set_ylabel("explained variance")
fig.tight_layout()
fig.savefig(out / "scree.png", dpi=140)
plt.close(fig)

mean = model.mean_.reshape(IMAGE_SIZE, IMAGE_SIZE)
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(mean, cmap="gray")
ax.set_title("mean image")
ax.axis("off")
fig.tight_layout()
fig.savefig(out / "mean.png", dpi=140)
plt.close(fig)

# Scale eigenfaces to 0-1 just for viewing.
faces = model.components_.copy()
for i in range(len(faces)):
    v = faces[i]
    faces[i] = (v - v.min()) / (v.max() - v.min() + 1e-8)
show_as_images(faces, "eigenfaces (first components)", out / "eigenfaces.png")
reconstructions(X_train, train_scores, model, train_paths, out / "train_reconstructions.png")
print(f"wrote {out / 'scree.png'}")
print(f"wrote {out / 'mean.png'}")
print(f"wrote {out / 'eigenfaces.png'}")
print(f"wrote {out / 'train_reconstructions.png'}")

test_paths = list_images(TEST_DIR)
if test_paths:
    if len(test_paths) < 1:
        raise SystemExit("No readable test images.")
    print(f"loading test from {TEST_DIR}")
    X_test, test_paths, test_labels = load_folder(TEST_DIR) if len(test_paths) >= 2 else (
        np.stack([load_one(test_paths[0])]),
        test_paths,
        [test_paths[0].parent.name if test_paths[0].parent != TEST_DIR else ""],
    )
    test_scores = model.transform(X_test)
    test_frame = pd.DataFrame(test_scores, columns=score_cols)
    test_frame.insert(0, "file", [p.name for p in test_paths])
    if any(test_labels):
        test_frame.insert(1, "label", test_labels)
    test_frame.to_csv(out / "test_scores.csv", index=False)
    reconstructions(X_test, test_scores, model, test_paths, out / "test_reconstructions.png")
    print(f"test: {len(test_paths)} photos")
    print(f"wrote {out / 'test_scores.csv'}")
    print(f"wrote {out / 'test_reconstructions.png'}")
else:
    print(f"no test photos in {TEST_DIR} — skipped test step")
