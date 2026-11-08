"""Run PCA + KNN and PCA + LDA + KNN experiments."""

# Machine learning libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Project modules
from src.lbph import LBPH_GRID_X, LBPH_GRID_Y, LBPH_NEIGHBORS, LBPH_RADIUS
from src.preprocessing import IMAGE_SIZE, N_FEATURES

# Fixed settings, identical for both datasets
PCA_VARIANCE_RETAINED = 0.95
KNN_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_pca_knn_pipeline():
    """Build pipeline 1: PCA -> KNN."""
    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE_RETAINED,
                    # Required when n_components is a variance fraction
                    svd_solver="full",
                    random_state=RANDOM_STATE,
                ),
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


def build_pca_lda_knn_pipeline():
    """Build pipeline 2: PCA -> LDA -> KNN."""
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
                # Default n_components: at most (identities - 1)
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
    """Split data by index: 80% train, 20% test, stratified."""
    # Indices are split once and reused by all three experiments
    positions = np.arange(len(y))
    return train_test_split(
        positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def run_single_experiment(name, pipeline, X_train, X_test, y_train, y_test):
    """Train one pipeline and evaluate it once on the test set."""
    # Train model
    pipeline.fit(X_train, y_train)

    # Predict test identities and calculate accuracy
    y_predicted = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)

    # Read the component counts chosen during training
    n_pca_components = pipeline.named_steps["pca"].n_components_
    n_lda_components = None
    if "lda" in pipeline.named_steps:
        n_lda_components = pipeline.named_steps["lda"].scalings_.shape[1]

    return {
        "name": name,
        "kind": "sklearn",
        "accuracy": accuracy,
        "y_predicted": y_predicted,
        "n_pca_components": n_pca_components,
        "n_lda_components": n_lda_components,
    }


def print_dataset_summary(dataset_name, X, y, X_train, X_test, results):
    """Print the results summary for one dataset."""
    identities = np.unique(y)

    # Dataset and preprocessing details
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

    # Settings and accuracy of each model
    for result in results:
        print()
        print("-" * 31)
        print(result["name"])
        print("-" * 31)
        print()

        if result.get("kind") == "lbph":
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
