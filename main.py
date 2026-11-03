"""
MAIN
====

Runs the SAME experiment, with the SAME settings, independently on each
dataset.

    run_experiment("datasets/dataset1")  ->  results/dataset1/
    run_experiment("datasets/dataset2")  ->  results/dataset2/

The two datasets are loaded, split, trained and evaluated in completely
separate function calls. No array is ever shared between them, so their
training and testing images can never mix.

Run it with:

    python main.py
"""

from pathlib import Path

from src.data_loader import load_dataset
from src.evaluation import (
    save_accuracy_comparison_chart,
    save_comparison_csv,
    save_confusion_matrix,
)
from src.experiments import (
    build_pca_knn_pipeline,
    build_pca_lda_knn_pipeline,
    print_dataset_summary,
    run_single_experiment,
    split_dataset,
)

# The datasets to run, in order. Each entry is processed on its own.
DATASETS = ["datasets/dataset1", "datasets/dataset2"]


def run_experiment(dataset_dir):
    """Run the complete experiment for ONE dataset."""
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    # Results for this dataset go in their own folder and nowhere else.
    output_dir = Path("results") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # STEP 1 - LOAD
    # In  : a folder of identity subfolders full of images.
    # Out : X (n_images, 4096) and y (n_images,) for THIS dataset only.
    # ---------------------------------------------------------------
    print(f"\nLoading {dataset_dir} ...")
    X, y = load_dataset(dataset_dir)

    # ---------------------------------------------------------------
    # STEP 2 - SPLIT
    # In  : all the images of this dataset.
    # Out : 80% training, 20% testing, stratified by identity.
    # The test set is now set aside and not touched until step 4.
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # ---------------------------------------------------------------
    # STEP 3 - TRAIN both pipelines on the training set only.
    # STEP 4 - EVALUATE each one exactly once on the untouched test set.
    # ---------------------------------------------------------------
    results = [
        run_single_experiment(
            "PCA + KNN",
            build_pca_knn_pipeline(),
            X_train,
            X_test,
            y_train,
            y_test,
        ),
        run_single_experiment(
            "PCA + LDA + KNN",
            build_pca_lda_knn_pipeline(),
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    ]

    # ---------------------------------------------------------------
    # STEP 5 - REPORT to the screen.
    # ---------------------------------------------------------------
    print_dataset_summary(dataset_name, X, y, X_train, X_test, results)

    # ---------------------------------------------------------------
    # STEP 6 - SAVE every output file inside this dataset's own folder.
    # ---------------------------------------------------------------
    save_comparison_csv(results, output_dir)
    save_accuracy_comparison_chart(results, dataset_name, output_dir)

    confusion_matrix_filenames = {
        "PCA + KNN": "confusion_matrix_pca_knn.png",
        "PCA + LDA + KNN": "confusion_matrix_pca_lda_knn.png",
    }
    for result in results:
        save_confusion_matrix(
            y_test,
            result["y_predicted"],
            f"{result['name']} - {dataset_name}",
            output_dir / confusion_matrix_filenames[result["name"]],
        )

    print()
    print(f"Saved results to: {output_dir.resolve()}")
    for file_path in sorted(output_dir.iterdir()):
        print(f"  - {file_path.name}")

    return results


def main():
    for dataset_dir in DATASETS:
        try:
            run_experiment(dataset_dir)
        except (FileNotFoundError, ValueError) as problem:
            # One empty or missing dataset must not stop the other one.
            print(f"\nSkipping {dataset_dir}: {problem}")

    print("\nDone.")


if __name__ == "__main__":
    main()
