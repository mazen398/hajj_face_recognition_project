# PCA on photos

This is computer-vision PCA. You drop pictures in a folder. The script flattens them, fits PCA on the training photos, then projects test photos onto those same axes.

| where | what |
| --- | --- |
| `data/train/` | put training photos here |
| `data/test/` | put test photos here (optional) |
| `run.py` | the file you run |
| `pca.py` | the model |
| `outputs/` | results after you run |

Supported types: `.jpg .jpeg .png .bmp .webp .tif`

## Click-by-click

1. In the left file list, open the `data` folder.
2. Open `train`.
3. Copy your training photos into `data/train`. Thousands is fine. Subfolders are fine too (the folder name becomes a label).
4. If you have a separate test set, open `data/test` and copy those photos there.
5. Open `run.py` only if you want to change size or component count. Defaults are `IMAGE_SIZE = 64` and `N_COMPONENTS = 50`.
6. Open a terminal in this project folder.
7. Run:

```bash
pip install -r requirements.txt
python run.py
```

8. When it finishes, open the `outputs` folder.

## What you will see

- `outputs/train_scores.csv` — each training photo as PCA numbers
- `outputs/test_scores.csv` — each test photo, only if you added test photos
- `outputs/mean.png` — average of the training photos
- `outputs/eigenfaces.png` — the main visual patterns PCA found
- `outputs/train_reconstructions.png` — original vs rebuilt training photos
- `outputs/test_reconstructions.png` — original vs rebuilt test photos
- `outputs/scree.png` — how much each component matters

Every photo is converted to grayscale and resized to 64x64 so thousands of images fit in memory. Raise `IMAGE_SIZE` in `run.py` if you want more detail (slower, heavier).
