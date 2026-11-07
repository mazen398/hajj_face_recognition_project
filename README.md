# Facial recognition experiments: PCA + KNN, PCA + LDA + KNN, and LBPH

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
    data_loader.py   folders of images  ->  X, X_lbph and y
    preprocessing.py one image file     ->  a 4096-value vector (PCA)
                                        ->  a 64x64 uint8 image (LBPH)
    experiments.py   the two sklearn pipelines, the split, the report
    lbph.py          the separate LBPH recognition branch
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

**Preprocessing**, identical for both datasets, using OpenCV only. Every image
is read, converted to grayscale and resized to 64 x 64. From that point there
are two paths:

| path | used by | output |
| --- | --- | --- |
| scale to 0-1, then flatten | PCA, LDA, KNN | 4096 float values |
| keep uint8, keep 2-D | LBPH | a 64 x 64 image |

LBPH needs the second path because it compares each pixel against its
neighbours and then splits the image into an 8 x 8 grid of cells. Normalising
to 0-1 or flattening would break both of those.

**Split**, done separately for each dataset: 80% training / 20% testing,
stratified by identity, `random_state=42`. The split is computed once over
image **indices** and then applied to both representations, so all three
experiments train and test on exactly the same photographs. There is no
validation set, and the test set is used exactly once at the end.

**Experiment 1** — `PCA -> KNN`
PCA keeps 95% of the variance, then KNN classifies with `n_neighbors=5` and
`weights="distance"`.

**Experiment 2** — `PCA -> LDA -> KNN`
PCA keeps 95% of the variance, LDA then uses the labels to separate identities,
and the same KNN settings classify.

Both are scikit-learn `Pipeline` objects, so PCA and LDA are fitted on the
training data only and then applied to the test data. That is what prevents
data leakage.

**Experiment 3** — `LBPH`, a completely separate branch
The grayscale 64 x 64 images go straight into OpenCV's
`cv2.face.LBPHFaceRecognizer_create()`. No PCA features, no LDA features, no
flattened vectors are ever passed to it. Identity names are mapped to integers
for training (`0 -> person01`, `1 -> person02`, ...) and mapped back after
prediction.

OpenCV returns a **distance**, not a probability: lower means a closer match.
That raw value is recorded for every test prediction and is deliberately not
converted into a percentage.

## Output files, per dataset

| file | contents |
| --- | --- |
| `final_model_comparison.csv` | one row per model with its test accuracy |
| `accuracy_comparison.png` | bar chart comparing all three models |
| `confusion_matrix_pca_knn.png` | confusion matrix for Experiment 1 |
| `confusion_matrix_pca_lda_knn.png` | confusion matrix for Experiment 2 |
| `confusion_matrix_lbph.png` | confusion matrix for Experiment 3 |

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
| LBPH radius | 1 |
| LBPH neighbors | 8 |
| LBPH grid_x | 8 |
| LBPH grid_y | 8 |

No hyperparameter searching is performed. Every value above is fixed so both
datasets and all three experiments are treated identically.

## OpenCV package

This project needs **`opencv-contrib-python`**, not `opencv-python`. The
`cv2.face` module that provides LBPH ships only in the contrib build. The two
packages overwrite each other, so install only one. If the plain package is
already present:

```bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-contrib-python
```
