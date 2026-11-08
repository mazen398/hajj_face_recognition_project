"""Load one dataset folder into features and identity labels.

Expected layout, where the folder name is the identity label:

    datasets/dataset1/person01/image1.png
    datasets/dataset1/person02/image1.png
"""

# Standard libraries
from pathlib import Path

# Machine learning libraries
import numpy as np

# Project modules
from src.preprocessing import preprocess_image, preprocess_image_lbph

# Accepted image file types
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# Minimum images per identity for a stratified split
MIN_IMAGES_PER_IDENTITY = 2


def load_dataset(dataset_dir, verbose=True):
    """Load and preprocess one dataset."""
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    # Each subfolder is one identity
    identity_folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    if not identity_folders:
        raise ValueError(
            f"No identity folders found inside {dataset_dir}.\n"
            "Expected one folder per person, for example:\n"
            f"  {dataset_dir / 'person01'}/image1.png"
        )

    features = []  # X for PCA / LDA / KNN
    lbph_images = []  # X_lbph for LBPH
    labels = []  # y
    skipped_identities = []

    for identity_folder in identity_folders:
        identity_name = identity_folder.name

        # Collect this person's image files
        image_paths = sorted(
            p
            for p in identity_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # Convert each image for PCA and for LBPH, keeping both lists aligned
        person_features = []
        person_lbph_images = []
        for image_path in image_paths:
            feature_vector = preprocess_image(image_path)
            lbph_image = preprocess_image_lbph(image_path)
            if feature_vector is None or lbph_image is None:
                continue
            person_features.append(feature_vector)
            person_lbph_images.append(lbph_image)

        # Skip identities with too few images to split
        if len(person_features) < MIN_IMAGES_PER_IDENTITY:
            skipped_identities.append((identity_name, len(person_features)))
            continue

        features.extend(person_features)
        lbph_images.extend(person_lbph_images)
        labels.extend([identity_name] * len(person_features))

    if not features:
        raise ValueError(f"No usable images were found inside {dataset_dir}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    # Report skipped identities
    if verbose and skipped_identities:
        print(f"  Skipped {len(skipped_identities)} identity folder(s) with fewer "
              f"than {MIN_IMAGES_PER_IDENTITY} readable images:")
        for name, count in skipped_identities[:10]:
            print(f"    - {name} ({count} image(s))")
        if len(skipped_identities) > 10:
            print(f"    ... and {len(skipped_identities) - 10} more")

    return X, lbph_images, y
