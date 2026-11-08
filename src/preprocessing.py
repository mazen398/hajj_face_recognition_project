"""Image preprocessing for the PCA and LBPH branches."""

# OpenCV and image processing
import cv2
import numpy as np

# Fixed image size and feature count
IMAGE_SIZE = 64
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE


def preprocess_image(image_path):
    """Convert one image into a 4096-value feature vector for PCA."""
    # Read image bytes (safe with non-English paths)
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if raw_bytes.size == 0:
        return None

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 64 x 64
    resized_image = cv2.resize(
        gray_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # Normalize 0-255 to 0-1
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Flatten into 4096 values
    feature_vector = normalized_image.flatten()

    return feature_vector


def preprocess_image_lbph(image_path):
    """Convert one image into a 64x64 uint8 image for LBPH."""
    # Read image bytes (safe with non-English paths)
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if raw_bytes.size == 0:
        return None

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 64 x 64
    resized_image = cv2.resize(
        gray_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # Keep 2-D and uint8: no normalization, no flattening
    return resized_image.astype(np.uint8)
