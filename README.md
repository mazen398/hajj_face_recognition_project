# Facial recognition experiment: PCA + KNN vs PCA + LDA + KNN

Two face datasets are tested **completely separately** using the same system and
the same fixed settings, so their results can be compared fairly.

## Project structure

```
datasets/
    dataset1/        put the first dataset here
    dataset2/        put the second dataset here
results/
    dataset1/        outputs for dataset1 only
    dataset2/        outputs for dataset2 only
src/
    data_loader.py   folders of images  ->  X (features) and y (labels)
    preprocessing.py one image file     ->  one 4096-value vector
    experiments.py   the two pipelines, the split, the training, the report
    evaluation.py    CSV, accuracy chart, confusion matrices
main.py              runs the whole experiment on each dataset in turn
requirements.txt
```

## Dataset layout

One folder per person. The folder name is the identity label.

```
datasets/dataset1/person01/image1.png
datasets/dataset1/person01/image2.png
datasets/dataset1/person02/image1.png
```

Each person needs at least 2 images so they can appear in both the training and
the testing split.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

## What the code does

**Preprocessing**, identical for both datasets, using OpenCV only:
read the image, convert to grayscale, resize to 64 x 64, scale pixels from
0-255 to 0-1, flatten into 4096 values.

**Split**, done separately for each dataset: 80% training / 20% testing,
stratified by identity, `random_state=42`. There is no validation set, and the
test set is used exactly once at the end.

**Experiment 1** — `PCA -> KNN`
PCA keeps 95% of the variance, then KNN classifies with `n_neighbors=5` and
`weights="distance"`.

**Experiment 2** — `PCA -> LDA -> KNN`
PCA keeps 95% of the variance, LDA then uses the labels to separate identities,
and the same KNN settings classify.

Both are scikit-learn `Pipeline` objects, so PCA and LDA are fitted on the
training data only and then applied to the test data. That is what prevents
data leakage.

## Output files, per dataset

| file | contents |
| --- | --- |
| `final_model_comparison.csv` | one row per model with its test accuracy |
| `accuracy_comparison.png` | bar chart comparing the two models |
| `confusion_matrix_pca_knn.png` | confusion matrix for Experiment 1 |
| `confusion_matrix_pca_lda_knn.png` | confusion matrix for Experiment 2 |

`results/dataset1/` and `results/dataset2/` are written independently and are
never mixed.

## Fixed settings

| setting | value |
| --- | --- |
| image size | 64 x 64 grayscale |
| features per image | 4096 |
| PCA variance retained | 95% |
| KNN neighbours (K) | 5 |
| KNN weighting | distance |
| train / test | 80% / 20%, stratified |
| random_state | 42 |

No hyperparameter searching is performed. Every value above is fixed so both
datasets and both experiments are treated identically.

## Not included yet

LBPH is deliberately left out. It will be added later as a separate
recognition branch.
