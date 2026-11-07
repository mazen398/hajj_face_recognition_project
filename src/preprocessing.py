"""
PREPROCESSING
=============

WHAT COMES IN : the path of one image file on disk.
WHAT HAPPENS  : the image is read, turned to grayscale, resized to 64x64,
                scaled to the 0-1 range, and flattened into one long row.
WHAT GOES OUT : a 1-D NumPy array of 4096 numbers (64 * 64).

WHY THIS STEP IS NECESSARY
--------------------------
PCA, LDA and KNN all expect every sample to be a row of numbers, and every row
must have the SAME length. Photographs do not satisfy that: they have different
widths, heights and colour. This step forces every photograph into one fixed
shape so the maths downstream is valid.

Each decision below matters:

  * Grayscale  - colour triples the number of features without helping identity
                 much, and PCA would spend components on lighting colour.
  * 64 x 64    - a fixed size is REQUIRED so every vector has 4096 values.
  * 0-1 scale  - KNN measures distances. Keeping all features on the same small
                 scale stops large pixel values from dominating the distance.
  * Flatten    - converts the 2-D image grid into the 1-D row scikit-learn wants.

The exact same function is used for dataset1 and dataset2, which is what makes
the comparison between the two datasets fair.
"""

import cv2
import numpy as np

# Every image becomes IMAGE_SIZE x IMAGE_SIZE pixels.
IMAGE_SIZE = 64

# 64 * 64 = 4096 values per image after flattening.
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE


def preprocess_image(image_path):
    """Turn one image file into a 4096-value feature vector.

    Returns None if the file cannot be read, so the caller can skip it
    instead of crashing the whole run.
    """
    # STEP 1 - READ THE IMAGE FROM DISK.
    # cv2.imread() cannot handle non-English characters in Windows paths, so we
    # read the raw bytes ourselves and let OpenCV decode them. Same result,
    # but it never silently fails on a folder name with unusual characters.
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if raw_bytes.size == 0:
        return None

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        # Not a readable image (corrupt file, wrong extension, and so on).
        return None

    # STEP 2 - CONVERT TO GRAYSCALE.
    # In  : a colour image with 3 channels (blue, green, red).
    # Out : a single-channel image, one brightness value per pixel.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # STEP 3 - RESIZE TO A FIXED 64 x 64.
    # INTER_AREA is the recommended interpolation when shrinking an image;
    # it averages the pixels being removed instead of just dropping them.
    resized_image = cv2.resize(
        gray_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # STEP 4 - NORMALISE FROM 0-255 TO 0-1.
    normalized_image = resized_image.astype(np.float32) / 255.0

    # STEP 5 - FLATTEN THE 64 x 64 GRID INTO ONE ROW OF 4096 VALUES.
    feature_vector = normalized_image.flatten()

    return feature_vector


def preprocess_image_lbph(image_path):
    """Turn one image file into a 64x64 grayscale image for LBPH.

    WHAT COMES IN : the path of one image file.
    WHAT GOES OUT : a 2-D NumPy array of shape (64, 64) holding uint8 values
                    in the normal 0-255 range. Returns None if unreadable.

    WHY THIS IS DIFFERENT FROM preprocess_image()
    ---------------------------------------------
    LBPH is not a vector method, so the last two steps above are deliberately
    skipped:

      * NOT normalised to 0-1. LBPH compares each pixel against its neighbours
        to build a binary pattern, and OpenCV's implementation expects 8-bit
        values. Float images in the 0-1 range are rejected.
      * NOT flattened. LBPH needs the 2-D grid because it divides the image
        into an 8 x 8 arrangement of cells and builds a histogram per cell.
        Flattening would destroy the spatial layout it depends on.

    The first three steps - read, grayscale, resize to 64 x 64 - are identical
    to the PCA path, so all three experiments see the same picture content at
    the same resolution.
    """
    # STEP 1 - READ THE IMAGE FROM DISK (same byte-safe read as above).
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if raw_bytes.size == 0:
        return None

    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # STEP 2 - CONVERT TO GRAYSCALE.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # STEP 3 - RESIZE TO A FIXED 64 x 64.
    resized_image = cv2.resize(
        gray_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # Stop here. Keep it 2-D and keep it uint8.
    return resized_image.astype(np.uint8)
