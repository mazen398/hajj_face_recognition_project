"""LBPH recognition branch, separate from the PCA and LDA pipelines."""

# OpenCV and image processing
import cv2
import numpy as np

# Fixed LBPH settings
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8


def check_lbph_available():
    """Check that cv2.face is installed."""
    if not hasattr(cv2, "face"):
        raise ImportError(
            "cv2.face is not available, so LBPH cannot run.\n"
            "LBPH lives in the OpenCV contrib build. Fix it with:\n\n"
            "    pip uninstall -y opencv-python opencv-python-headless\n"
            "    pip install opencv-contrib-python\n"
        )
    return True


def build_label_mapping(identity_names):
    """Map identity names to integer labels, and back."""
    unique_names = sorted(set(identity_names))
    name_to_id = {name: index for index, name in enumerate(unique_names)}
    id_to_name = {index: name for name, index in name_to_id.items()}
    return name_to_id, id_to_name


def create_lbph_recognizer():
    """Create the OpenCV LBPH recognizer."""
    check_lbph_available()
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )


def run_lbph_experiment(train_images, y_train, test_images, y_test):
    """Train LBPH and evaluate it once on the test set."""
    recognizer = create_lbph_recognizer()

    # Map identity names to integer labels
    name_to_id, id_to_name = build_label_mapping(y_train)
    train_label_ids = np.array([name_to_id[name] for name in y_train], dtype=np.int32)

    # Train model
    recognizer.train(train_images, train_label_ids)

    # Predict test identities and record the raw distance (lower is closer)
    predicted_names = []
    distances = []
    for test_image in test_images:
        predicted_id, distance = recognizer.predict(test_image)
        predicted_names.append(id_to_name.get(predicted_id, "unknown"))
        distances.append(distance)

    predicted_names = np.asarray(predicted_names)
    distances = np.asarray(distances, dtype=np.float64)

    # Calculate accuracy
    accuracy = float(np.mean(predicted_names == np.asarray(y_test)))

    return {
        "name": "LBPH",
        "kind": "lbph",
        "accuracy": accuracy,
        "y_predicted": predicted_names,
        "distances": distances,
        "n_pca_components": None,
        "n_lda_components": None,
    }
