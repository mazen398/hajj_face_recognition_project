# PCA face recognition (eigenfaces)

One file: `pca_photos.py`. It learns eigenfaces from training photos, recognises test photos it has never seen, and reports **accuracy**.

## Important for accuracy

Accuracy only exists when there is more than one person to tell apart. So point at the folder that **contains one subfolder per person**, not at a single person:

```python
# right - gives accuracy
PHOTOS_FOLDER = r"C:\Users\darks\Desktop\data_set\105_classes_pins_dataset"

# wrong - only one person, nothing to classify
PHOTOS_FOLDER = r"C:\...\105_classes_pins_dataset\pins_Chris Pratt"
```

## Run it

```bash
pip install numpy matplotlib pillow
python pca_photos.py
```

## Where everything is, for your report

Everything lands in a folder named **`pca_results`**, created right next to `pca_photos.py`.

| file | what to say about it in your report |
| --- | --- |
| `RESULTS.txt` | every number in one place: accuracy, splits, per-person scores. Screenshot this first. |
| `test_predictions.png` | test faces with CORRECT in green and WRONG in red |
| `accuracy_vs_eigenfaces.png` | accuracy rising as you keep more components, then flattening |
| `eigenfaces.png` | the principal components drawn as faces |
| `average_face.png` | the mean face PCA subtracts before fitting |
| `how_much_each_pattern_matters.png` | the scree plot |
| `test_rebuilt.png` | original vs reconstruction from components only |
| `test_predictions.csv` | per-photo: true person, predicted person, correct flag, PCA values |
| `train_scores.csv` | every training photo as PCA numbers |

## What the numbers mean

- **Accuracy** — share of unseen test photos matched to the right person.
- **Random guessing** — printed next to accuracy so you can show the model beats chance.
- **Detail retained** — how much of the image variation the kept eigenfaces preserve.
- **Compression** — 4096 pixels reduced to roughly 100 numbers per photo. That is the point of PCA.

## Method, if you need to write it up

1. Every photo is converted to grayscale and resized to 64x64, giving a 4096-value vector.
2. Photos are split per person, so each person appears in both train and test.
3. PCA is fit **only** on the training set. The test set is never seen during fitting.
4. Both sets are projected onto the top components (the eigenfaces).
5. Each test face is labelled with its nearest training face in that reduced space (1-nearest-neighbour, Euclidean distance).

## Settings you can change

```python
IMAGE_SIZE = 64       # bigger = more detail, slower
N_COMPONENTS = 100    # how many eigenfaces to keep
TEST_SHARE = 0.25     # held-out share per person
MAX_PEOPLE = 20       # raise toward 105 for the full dataset
MAX_PER_PERSON = 40   # cap per person, keeps the run fast
```

Changing `MAX_PEOPLE` and `N_COMPONENTS` and recording how accuracy moves makes a good results section.
