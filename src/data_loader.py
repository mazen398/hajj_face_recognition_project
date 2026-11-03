"""
DATA LOADER
===========

WHAT COMES IN : the path of ONE dataset folder, for example datasets/dataset1.
WHAT HAPPENS  : every person's folder is visited, every image inside it is
                preprocessed, and the folder name is recorded as that image's
                identity label.
WHAT GOES OUT : X  - a 2-D array of shape (number of images, 4096)
                y  - a 1-D array of identity labels, one per image

WHY THIS STEP IS NECESSARY
--------------------------
scikit-learn never looks at files. It only accepts a feature matrix X and a
label vector y. This file is the bridge between "folders of pictures on disk"
and "the two arrays scikit-learn needs".

EXPECTED FOLDER LAYOUT
----------------------
    datasets/dataset1/person01/image1.png
    datasets/dataset1/person01/image2.png
    datasets/dataset1/person02/image1.png

The folder name (person01, person02, ...) IS the class label.

IMPORTANT: this function is always called on ONE dataset at a time. dataset1 and
dataset2 are never loaded into the same arrays, so they can never be mixed.
"""

from pathlib import Path

import numpy as np

from src.preprocessing import preprocess_image

# File types we are willing to read.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

# A class needs at least 2 images: one can go to training and one to testing.
# With fewer than this, a stratified 80/20 split is impossible.
MIN_IMAGES_PER_IDENTITY = 2


def load_dataset(dataset_dir, verbose=True):
    """Load one dataset folder into X (features) and y (labels)."""
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    # Each SUBFOLDER of the dataset folder is one identity / person / class.
    identity_folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    if not identity_folders:
        raise ValueError(
            f"No identity folders found inside {dataset_dir}.\n"
            "Expected one folder per person, for example:\n"
            f"  {dataset_dir / 'person01'}/image1.png"
        )

    features = []  # will become X
    labels = []  # will become y
    skipped_identities = []  # folders with too few usable images

    for identity_folder in identity_folders:
        identity_name = identity_folder.name

        # Collect every image file belonging to this person.
        image_paths = sorted(
            p
            for p in identity_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # Preprocess each image into a 4096-value row.
        person_features = []
        for image_path in image_paths:
            feature_vector = preprocess_image(image_path)
            if feature_vector is not None:
                person_features.append(feature_vector)

        # A person with only one usable image cannot appear in both the
        # training and the testing set, so a stratified split would fail.
        # We skip them and report it rather than crashing.
        if len(person_features) < MIN_IMAGES_PER_IDENTITY:
            skipped_identities.append((identity_name, len(person_features)))
            continue

        features.extend(person_features)
        labels.extend([identity_name] * len(person_features))

    if not features:
        raise ValueError(f"No usable images were found inside {dataset_dir}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    if verbose and skipped_identities:
        print(f"  Skipped {len(skipped_identities)} identity folder(s) with fewer "
              f"than {MIN_IMAGES_PER_IDENTITY} readable images:")
        for name, count in skipped_identities[:10]:
            print(f"    - {name} ({count} image(s))")
        if len(skipped_identities) > 10:
            print(f"    ... and {len(skipped_identities) - 10} more")

    return X, y
