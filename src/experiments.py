"""
EXPERIMENTS
===========

This file defines the two recognition pipelines and runs the complete
experiment for ONE dataset.

EXPERIMENT 1 : face features -> PCA -> KNN -> predicted identity
EXPERIMENT 2 : face features -> PCA -> LDA -> KNN -> predicted identity

WHY A scikit-learn Pipeline
---------------------------
A Pipeline chains the steps together and guarantees the rule that matters most
in an experiment: every step is LEARNED from the training data only.

When we call pipeline.fit(X_train, y_train):
    PCA learns its components from X_train only.
    LDA learns its directions from the PCA output of X_train only.
    KNN memorises the transformed training samples only.

When we call pipeline.predict(X_test):
    the ALREADY-LEARNED PCA and LDA are applied to X_test.
    Nothing is re-fitted, so the test set never influences the model.

That is what "avoiding data leakage" means in practice. Doing the same thing
by hand is easy to get wrong, which is exactly why Pipeline exists.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.lbph import LBPH_GRID_X, LBPH_GRID_Y, LBPH_NEIGHBORS, LBPH_RADIUS
from src.preprocessing import IMAGE_SIZE, N_FEATURES

# ---------------------------------------------------------------------------
# FIXED SETTINGS
# These are deliberately constants, not search spaces. The project brief says
# no hyperparameter searching, so every dataset and every experiment uses the
# identical configuration. That is what makes the comparison fair.
# ---------------------------------------------------------------------------

PCA_VARIANCE_RETAINED = 0.95  # keep 95% of the variance
KNN_NEIGHBORS = 5  # K = 5
KNN_WEIGHTS = "distance"  # closer neighbours count more
TEST_SIZE = 0.20  # 20% held out for testing
RANDOM_STATE = 42  # fixed, so every run gives the same split


def build_pca_knn_pipeline():
    """EXPERIMENT 1 pipeline: PCA -> KNN.

    IN  : 4096 raw pixel features per image.
    PCA : compresses those 4096 correlated features down to however many
          components are needed to retain 95% of the variance. Neighbouring
          pixels in a face are highly correlated, so most of the 4096
          dimensions carry almost no new information. Removing them makes the
          distances KNN computes far more meaningful and much faster.
    KNN : stores the compressed training faces. A new face is labelled with the
          identity of the closest training faces.
    OUT : a predicted identity.
    """
    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE_RETAINED,
                    # svd_solver="full" is required when n_components is given
                    # as a variance fraction rather than a whole number.
                    svd_solver="full",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=KNN_NEIGHBORS,
                    # "distance" means a neighbour's vote is divided by its
                    # distance, so a face that is very close to the test face
                    # has more influence than one that is far away.
                    weights=KNN_WEIGHTS,
                ),
            ),
        ]
    )


def build_pca_lda_knn_pipeline():
    """EXPERIMENT 2 pipeline: PCA -> LDA -> KNN.

    IN  : 4096 raw pixel features per image.
    PCA : same compression as Experiment 1, keeping 95% of the variance.
          PCA is UNSUPERVISED. It keeps the directions where the images vary
          the most, which is often lighting and pose rather than identity.
    LDA : SUPERVISED. It uses the identity labels to find directions that push
          DIFFERENT people apart while pulling photos of the SAME person
          together. This is the step that should make identities easier to
          separate than PCA alone.
    KNN : final classification in that better-separated space.
    OUT : a predicted identity.

    Why PCA before LDA? LDA struggles when there are far more features than
    samples, which is exactly the case with 4096 raw pixels. Running PCA first
    reduces the dimensions to a manageable number, so LDA becomes stable.
    This two-stage combination is the classic "Fisherfaces" method.
    """
    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE_RETAINED,
                    svd_solver="full",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "lda",
                # n_components is left at its default. LDA can produce at most
                # (number of identities - 1) components, and scikit-learn picks
                # that maximum automatically.
                LinearDiscriminantAnalysis(),
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=KNN_NEIGHBORS,
                    weights=KNN_WEIGHTS,
                ),
            ),
        ]
    )


def split_indices(y):
    """Split ONE dataset into 80% training and 20% testing, by INDEX.

    IN  : y, the identity label of every image in this dataset.
    OUT : train_idx, test_idx - two arrays of positions into the dataset.

    WHY INDICES RATHER THAN THE ARRAYS THEMSELVES
    ---------------------------------------------
    Every image has two representations: a 4096-value vector for PCA and a
    64x64 image for LBPH. Splitting the positions once and then applying those
    same positions to both representations guarantees that all three
    experiments train on the identical photographs and are tested on the
    identical photographs. Splitting each representation separately would risk
    them drifting apart and would make the comparison meaningless.

    stratify=y keeps the proportion of each identity roughly the same in both
    halves, so no person ends up entirely in the training set or entirely in
    the test set.

    random_state=42 fixes the shuffling, so the split is identical every time
    the script runs. That is what makes the results reproducible.

    There is deliberately NO validation set: the brief asks for a single final
    evaluation on the untouched test set.
    """
    positions = np.arange(len(y))
    return train_test_split(
        positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def run_single_experiment(name, pipeline, X_train, X_test, y_train, y_test):
    """Train one pipeline on the training set and score it once on the test set.

    Returns a dictionary with the accuracy, the predictions and the sizes of
    the internal PCA / LDA steps so they can be printed and saved.
    """
    # TRAINING. Only X_train and y_train are visible here.
    pipeline.fit(X_train, y_train)

    # FINAL EVALUATION. The test set is used exactly once, right here.
    y_predicted = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)

    # Pull the learned sizes out of the fitted pipeline for the report.
    n_pca_components = pipeline.named_steps["pca"].n_components_
    n_lda_components = None
    if "lda" in pipeline.named_steps:
        # scalings_ has one column per discriminant direction LDA kept.
        n_lda_components = pipeline.named_steps["lda"].scalings_.shape[1]

    return {
        "name": name,
        # "kind" tells the report which block of settings to print.
        # LBPH results are built in src/lbph.py and use kind="lbph".
        "kind": "sklearn",
        "accuracy": accuracy,
        "y_predicted": y_predicted,
        "n_pca_components": n_pca_components,
        "n_lda_components": n_lda_components,
    }


def print_dataset_summary(dataset_name, X, y, X_train, X_test, results):
    """Print the readable report for one dataset."""
    identities = np.unique(y)

    print()
    print("=" * 32)
    print(f"DATASET: {dataset_name}")
    print("=" * 32)
    print()
    print(f"Total images:         {len(X)}")
    print(f"Number of identities: {len(identities)}")
    print(f"Training images:      {len(X_train)}")
    print(f"Testing images:       {len(X_test)}")
    print()
    print("IMAGE PREPROCESSING")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Original features per image: {N_FEATURES}")

    for result in results:
        print()
        print("-" * 31)
        print(result["name"])
        print("-" * 31)
        print()

        if result.get("kind") == "lbph":
            # LBPH has no PCA or LDA stage, so it reports its own settings.
            print(f"Radius: {LBPH_RADIUS}")
            print(f"Neighbors: {LBPH_NEIGHBORS}")
            print(f"Grid X: {LBPH_GRID_X}")
            print(f"Grid Y: {LBPH_GRID_Y}")
        else:
            print(f"PCA variance retained: {int(PCA_VARIANCE_RETAINED * 100)}%")
            print(f"Number of PCA components selected: {result['n_pca_components']}")
            if result["n_lda_components"] is not None:
                print(f"Number of LDA components: {result['n_lda_components']}")
            print(f"K: {KNN_NEIGHBORS}")
            print(f"Weighting: {KNN_WEIGHTS}")

        print(f"Test accuracy: {result['accuracy']:.4f}  ({result['accuracy']:.2%})")
