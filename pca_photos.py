"""
Face recognition with PCA (eigenfaces).

WHAT IT DOES
  1. Reads your photos.
  2. Splits them into a training set and a test set.
  3. Learns "eigenfaces" from the training set only.
  4. Recognises the test faces and reports ACCURACY.

HOW TO USE
  Set PHOTOS_FOLDER below, then run:  python pca_photos.py
  Everything is written to the "pca_results" folder next to this file.

To EDIT this file, right-click it and choose Open with > Notepad.
Double-clicking it RUNS it instead of opening it.
"""

import traceback
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ================= CHANGE THIS LINE =================
# Point at the folder that holds ONE SUBFOLDER PER PERSON.
# That is what gives you an accuracy number.
PHOTOS_FOLDER = r"C:\Users\darks\Desktop\data_set\105_classes_pins_dataset"
# ====================================================

IMAGE_SIZE = 64  # each photo is resized to 64x64
N_COMPONENTS = 100  # how many eigenfaces to keep
TEST_SHARE = 0.25  # share of each person's photos held back for testing
MAX_PEOPLE = 20  # use the first N people. Raise it if you want the full 105.
MAX_PER_PERSON = 40  # cap per person, keeps the run fast

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RNG = np.random.default_rng(0)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        var = (s**2) / (X.shape[0] - 1)
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores):
        return scores @ self.components_ + self.mean_


def read_image(path):
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(img, dtype=float).reshape(-1) / 255.0


def collect(folder):
    """Return X, labels, paths. Each subfolder of `folder` is one person."""
    if not folder.exists():
        raise SystemExit(
            f"\nThat folder does not exist:\n  {folder}\n\n"
            "Open this file in Notepad and fix the PHOTOS_FOLDER line near the top."
        )

    people = sorted(p for p in folder.iterdir() if p.is_dir())
    if not people:
        people = [folder]  # a single flat folder of photos
    people = people[:MAX_PEOPLE]

    X, labels, paths = [], [], []
    for person in people:
        shots = sorted(p for p in person.rglob("*") if p.suffix.lower() in EXTS)
        if len(shots) < 4:
            continue
        if len(shots) > MAX_PER_PERSON:
            shots = [shots[i] for i in RNG.permutation(len(shots))[:MAX_PER_PERSON]]
        name = person.name.replace("pins_", "")
        for shot in shots:
            try:
                X.append(read_image(shot))
            except Exception:
                continue
            labels.append(name)
            paths.append(shot)
        print(f"  {name}: {len(shots)} photos")

    if len(X) < 8:
        raise SystemExit(f"\nOnly found {len(X)} usable photos in:\n  {folder}")
    return np.stack(X), np.array(labels), paths


def split_per_person(labels):
    """Hold back TEST_SHARE of EACH person, so every person is in both sets."""
    train_idx, test_idx = [], []
    by_person = defaultdict(list)
    for i, name in enumerate(labels):
        by_person[name].append(i)
    for idxs in by_person.values():
        idxs = [idxs[i] for i in RNG.permutation(len(idxs))]
        cut = max(1, int(round(len(idxs) * (1 - TEST_SHARE))))
        train_idx += idxs[:cut]
        test_idx += idxs[cut:]
    return np.array(train_idx), np.array(test_idx)


def nearest_neighbour(train_scores, train_labels, test_scores):
    """Classify each test face as the closest training face in PCA space."""
    predictions = []
    for row in test_scores:
        distances = np.linalg.norm(train_scores - row, axis=1)
        predictions.append(train_labels[int(np.argmin(distances))])
    return np.array(predictions)


def save_grid(images, title, path, n=16):
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


def save_predictions(X_test, truth, predicted, path, n=12):
    n = min(n, len(X_test))
    cols = 4
    rows = int(np.ceil(n / cols))
    # Mix the order so the grid shows several different people, and make sure
    # a few mistakes are visible rather than only easy hits.
    wrong = [i for i in range(len(truth)) if truth[i] != predicted[i]]
    right_ones = [i for i in range(len(truth)) if truth[i] == predicted[i]]
    pick = wrong[: max(2, n // 4)] + list(RNG.permutation(right_ones))
    pick = pick[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.9 * rows))
    for slot, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if slot >= len(pick):
            continue
        i = pick[slot]
        ax.imshow(X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
        hit = truth[i] == predicted[i]
        ax.set_title(
            f"{'CORRECT' if hit else 'WRONG'}\nis: {truth[i]}\nsaid: {predicted[i]}",
            fontsize=7,
            color="green" if hit else "red",
        )
    fig.suptitle("test faces the model had never seen")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_rebuilds(X, rebuilt, path, n=6):
    n = min(n, len(X))
    fig, axes = plt.subplots(n, 2, figsize=(4.5, 2 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[i, 0].imshow(X[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(rebuilt[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("original", fontsize=8)
        axes[i, 1].set_title("rebuilt from eigenfaces", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    folder = Path(PHOTOS_FOLDER).expanduser()
    out = Path(__file__).parent / "pca_results"
    out.mkdir(exist_ok=True)
    report = []

    def say(line=""):
        print(line)
        report.append(line)

    print(f"Reading photos from: {folder}\n")
    X, labels, paths = collect(folder)
    people = sorted(set(labels))

    train_idx, test_idx = split_per_person(labels)
    X_train, y_train = X[train_idx], labels[train_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    say("=" * 52)
    say("            PCA FACE RECOGNITION RESULTS")
    say("=" * 52)
    say(f"people          : {len(people)}")
    say(f"photos total    : {len(X)}")
    say(f"training photos : {len(X_train)}")
    say(f"test photos     : {len(X_test)}")
    say(f"image size      : {IMAGE_SIZE} x {IMAGE_SIZE} grayscale")
    say(f"pixels per photo: {IMAGE_SIZE * IMAGE_SIZE}")

    k = min(N_COMPONENTS, len(X_train) - 1)
    model = PCA(k).fit(X_train)
    train_scores = model.transform(X_train)
    test_scores = model.transform(X_test)

    say(f"eigenfaces kept : {k}")
    say(f"compression     : {IMAGE_SIZE * IMAGE_SIZE} numbers -> {k} numbers per photo")
    say(f"detail retained : {model.explained_variance_ratio_.sum():.1%}")
    say()

    predicted = nearest_neighbour(train_scores, y_train, test_scores)
    correct = predicted == y_test
    accuracy = correct.mean()

    say("-" * 52)
    say(f"ACCURACY ON TEST PHOTOS : {accuracy:.1%}   ({correct.sum()}/{len(correct)})")
    say(f"random guessing would be: {1 / len(people):.1%}")
    say("-" * 52)
    say()

    say("accuracy per person:")
    for name in people:
        mask = y_test == name
        if mask.sum():
            say(f"  {name:<28} {correct[mask].mean():6.1%}  ({correct[mask].sum()}/{mask.sum()})")
    say()

    say("variance held by the first eigenfaces:")
    running = 0.0
    for i, r in enumerate(model.explained_variance_ratio_[:10], start=1):
        running += r
        say(f"  eigenface {i:>2}: {r:6.1%}   running total {running:.1%}")

    # Accuracy as the number of eigenfaces grows.
    say()
    say("accuracy vs number of eigenfaces:")
    ks, accs = [], []
    for test_k in [5, 10, 20, 40, 60, 80, 100, 150, 200]:
        if test_k > k:
            break
        a = (nearest_neighbour(train_scores[:, :test_k], y_train, test_scores[:, :test_k]) == y_test).mean()
        ks.append(test_k)
        accs.append(a)
        say(f"  {test_k:>3} eigenfaces : {a:.1%}")

    (out / "RESULTS.txt").write_text("\n".join(report), encoding="utf-8")

    header = "file,person,predicted,correct," + ",".join(f"PC{i+1}" for i in range(k))
    with open(out / "test_predictions.csv", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, idx in enumerate(test_idx):
            f.write(
                f"{paths[idx].name},{y_test[i]},{predicted[i]},{int(correct[i])},"
                + ",".join(f"{v:.5f}" for v in test_scores[i])
                + "\n"
            )
    with open(out / "train_scores.csv", "w", encoding="utf-8") as f:
        f.write("file,person," + ",".join(f"PC{i+1}" for i in range(k)) + "\n")
        for i, idx in enumerate(train_idx):
            f.write(
                f"{paths[idx].name},{y_train[i]},"
                + ",".join(f"{v:.5f}" for v in train_scores[i])
                + "\n"
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, accs, marker="o")
    ax.set_xlabel("number of eigenfaces")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title("accuracy vs number of eigenfaces")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_eigenfaces.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(1, k + 1), model.explained_variance_ratio_)
    ax.set_xlabel("eigenface number")
    ax.set_ylabel("share of detail")
    ax.set_title("how much each eigenface matters")
    fig.tight_layout()
    fig.savefig(out / "how_much_each_pattern_matters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(model.mean_.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    ax.set_title("average face")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "average_face.png", dpi=140)
    plt.close(fig)

    faces = model.components_.copy()
    for i, v in enumerate(faces):
        faces[i] = (v - v.min()) / (v.max() - v.min() + 1e-8)
    save_grid(faces, "eigenfaces", out / "eigenfaces.png")

    save_rebuilds(X_test, model.inverse_transform(test_scores), out / "test_rebuilt.png")
    save_predictions(X_test, y_test, predicted, out / "test_predictions.png")

    say()
    say(f"Everything saved in: {out}")
    (out / "RESULTS.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nOpen this folder:\n  {out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:
        print(stop)
    except Exception:
        traceback.print_exc()
    # Keeps the black window open when the file is launched by double-click.
    input("\nPress Enter to close this window...")
