"""Run the facial recognition experiments separately on each dataset."""

# Standard libraries
from pathlib import Path

# Project modules
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
    split_indices,
)
from src.lbph import run_lbph_experiment

# Datasets to run, each one independently
DATASETS = ["datasets/dataset1", "datasets/dataset2"]


def run_experiment(dataset_dir):
    """Run all three experiments on one dataset."""
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name

    # Results folder for this dataset only
    output_dir = Path("results") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading {dataset_dir} ...")
    X, X_lbph, y = load_dataset(dataset_dir)

    # Split data: 80% train, 20% test
    train_idx, test_idx = split_indices(y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply the same split to the LBPH images
    lbph_train_images = [X_lbph[i] for i in train_idx]
    lbph_test_images = [X_lbph[i] for i in test_idx]

    # Train and evaluate the three models
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
        run_lbph_experiment(
            lbph_train_images,
            y_train,
            lbph_test_images,
            y_test,
        ),
    ]

    # Print summary
    print_dataset_summary(dataset_name, X, y, X_train, X_test, results)

    # Save CSV and comparison graph
    save_comparison_csv(results, output_dir)
    save_accuracy_comparison_chart(results, dataset_name, output_dir)

    # Create confusion matrix for each model
    confusion_matrix_filenames = {
        "PCA + KNN": "confusion_matrix_pca_knn.png",
        "PCA + LDA + KNN": "confusion_matrix_pca_lda_knn.png",
        "LBPH": "confusion_matrix_lbph.png",
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
            # Skip a missing or empty dataset without stopping the other
            print(f"\nSkipping {dataset_dir}: {problem}")

    print("\nDone.")


if __name__ == "__main__":
    main()
