"""
LBPH  -  Local Binary Patterns Histograms
=========================================

This is EXPERIMENT 3, and it is a completely separate recognition branch from
the PCA / LDA / KNN pipelines.

    Face image -> grayscale -> resize 64x64 -> LBPH -> predicted identity

WHAT MAKES LBPH DIFFERENT
-------------------------
PCA and LDA are global methods. They look at the whole face as one long vector
and ask "which directions in this 4096-dimensional space matter most?".

LBPH is a LOCAL texture method. For every pixel it compares the brightness of
the surrounding ring of neighbours against the centre pixel, producing an
8-bit binary pattern per pixel. The image is then divided into an 8 x 8 grid of
cells, a histogram of those patterns is built for each cell, and all 64
histograms are joined into one long descriptor. Recognition is then a
histogram comparison.

Because of that, LBPH must NEVER receive PCA features, LDA features, or a
flattened vector. It needs the raw 2-D grayscale grid, in 0-255 uint8, so the
pixel-versus-neighbour comparisons and the spatial grid are meaningful.

ABOUT THE RETURNED VALUE
------------------------
OpenCV's predict() returns (label, distance). The second number is a DISTANCE,
not a probability:

    lower  = the test face closely matches a trained face
    higher = the match is poor

It is deliberately stored raw and never converted into a percentage.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# FIXED LBPH SETTINGS
# These are OpenCV's standard defaults. No hyperparameter searching is done,
# which matches how the KNN settings are fixed in the other experiments.
# ---------------------------------------------------------------------------

LBPH_RADIUS = 1  # distance from the centre pixel to its neighbours
LBPH_NEIGHBORS = 8 # how many neighbours are sampled on that ring
LBPH_GRID_X = 8  # the image is divided into 8 cells horizontally
LBPH_GRID_Y = 8  # and 8 cells vertically, so 64 histograms in total


def check_lbph_available():
    """Confirm cv2.face exists, with a clear message if it does not.

    cv2.face ships only in the contrib build of OpenCV. If the plain
    opencv-python package is installed instead, the attribute is missing and
    the error would otherwise be cryptic.
    """
    if not hasattr(cv2, "face"):
        raise ImportError(
            "cv2.face is not available, so LBPH cannot run.\n"
            "LBPH lives in the OpenCV contrib build. Fix it with:\n\n"
            "    pip uninstall -y opencv-python opencv-python-headless\n"
            "    pip install opencv-contrib-python\n"
        )
    return True


def build_label_mapping(identity_names):
    """Map identity names to integers, because OpenCV trains on integers.

    IN  : the array of identity names, for example
          ["yaleB01", "yaleB01", "yaleB02", ...]
    OUT : name_to_id  {"yaleB01": 0, "yaleB02": 1, "yaleB03": 2, ...}
          id_to_name  {0: "yaleB01", 1: "yaleB02", 2: "yaleB03", ...}

    OpenCV's recognizer cannot store text labels, so every identity gets a
    number. The reverse mapping is kept so predictions can be turned back into
    the original folder names, which is what the confusion matrix needs.

    The names are sorted first so the numbering is stable between runs.
    """
    unique_names = sorted(set(identity_names))
    name_to_id = {name: index for index, name in enumerate(unique_names)}
    id_to_name = {index: name for name, index in name_to_id.items()}
    return name_to_id, id_to_name


def create_lbph_recognizer():
    """Create the OpenCV LBPH recogniser with the fixed settings above."""
    check_lbph_available()
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )


def run_lbph_experiment(train_images, y_train, test_images, y_test):
    """Train LBPH on the training images and score it once on the test images.

    IN  : train_images - list of 64x64 uint8 images (the 80% training split)
          y_train      - their identity names
          test_images  - list of 64x64 uint8 images (the untouched 20%)
          y_test       - their true identity names

    OUT : a result dictionary in the same shape the other experiments return,
          plus the raw LBPH distance recorded for every test prediction.
    """
    recognizer = create_lbph_recognizer()

    # STEP 1 - TURN IDENTITY NAMES INTO INTEGERS.
    # The mapping is built from the TRAINING names only, because those are the
    # only identities the recogniser will ever know about.
    name_to_id, id_to_name = build_label_mapping(y_train)
    train_label_ids = np.array([name_to_id[name] for name in y_train], dtype=np.int32)

    # STEP 2 - TRAIN. Only the training images are passed in.
    # OpenCV builds the LBP histograms for each training face and stores them
    # together with its integer label.
    recognizer.train(train_images, train_label_ids)

    # STEP 3 - PREDICT EVERY TEST IMAGE, once.
    predicted_names = []
    distances = []
    for test_image in test_images:
        predicted_id, distance = recognizer.predict(test_image)
        # An unknown id should not happen, but guard anyway so a surprise
        # never crashes the run.
        predicted_names.append(id_to_name.get(predicted_id, "unknown"))
        distances.append(distance)

    predicted_names = np.asarray(predicted_names)
    distances = np.asarray(distances, dtype=np.float64)

    # STEP 4 - ACCURACY, using the same definition as the other experiments:
    # the share of test images whose predicted identity is the true identity.
    accuracy = float(np.mean(predicted_names == np.asarray(y_test)))

    return {
        "name": "LBPH",
        "kind": "lbph",
        "accuracy": accuracy,
        "y_predicted": predicted_names,
        # Raw OpenCV distances, left exactly as returned. Lower means closer.
        "distances": distances,
        "n_pca_components": None,
        "n_lda_components": None,
    }
