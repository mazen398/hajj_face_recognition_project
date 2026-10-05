# PCA on photos — one file

`pca_photos.py` is the whole thing. You do **not** move your photos anywhere. You tell the file where your photos already live.

## The important part

This code is running on a **cloud computer**, not on your laptop. Your photos are on **your** laptop. They cannot see each other. So you run the file on your own machine.

## Steps on Windows

**1. Make a folder for the code**

Desktop, right-click, New > Folder. Name it `pca`.

**2. Get the file into it**

Open `pca_photos.py` in Cursor, select all, copy. On your Desktop make a new file `C:\Users\darks\Desktop\pca\pca_photos.py`, paste, save.

**3. Find your photo folder**

Open the folder with your photos in File Explorer. Click the address bar at the top. It shows something like:

```
C:\Users\darks\Pictures\dataset
```

Copy that.

**4. Put that path in the file**

Open `pca_photos.py` and change the one marked line:

```python
PHOTOS_FOLDER = r"C:\Users\darks\Pictures\dataset"
```

Keep the `r` and the quotes. Change nothing else.

**5. Install the libraries, once**

Open a terminal in that folder and run:

```bash
pip install numpy matplotlib pillow
```

**6. Run it**

```bash
python pca_photos.py
```

**7. Look at the results**

A folder called `pca_results` appears next to the file. Open it.

## What you get

| file | what it means |
| --- | --- |
| `average_photo.png` | the average of all your photos |
| `patterns_learned.png` | the main visual patterns PCA found |
| `train_before_after.png` | your photo next to the PCA rebuild |
| `test_before_after.png` | same, on photos the model never saw |
| `train_scores.csv` | every training photo turned into numbers |
| `test_scores.csv` | held-out photos, plus a rebuild error per photo |
| `how_much_each_pattern_matters.png` | which patterns carry the detail |

## Train and test

You don't split anything by hand. The file takes the **first 80%** of your photos to learn from and **holds back the last 20%** to test on. Change `TEST_SHARE` if you want a different split.

## Knobs, if you want them

```python
IMAGE_SIZE = 64      # bigger = more detail, slower
N_COMPONENTS = 50    # how many patterns to learn
TEST_SHARE = 0.2     # held-out share
MAX_PHOTOS = 4000    # cap, so thousands of photos still fit in memory
```
