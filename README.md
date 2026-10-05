# PCA

This folder is only three things:

| file | what it is |
| --- | --- |
| `pca.py` | the model |
| `run.py` | the file you edit and run |
| `data/` | where you put your tables |

## What to do

1. Copy your training file into `data/` (csv, tsv, or xlsx).
2. Open `run.py`.
3. Change these lines at the top:

```python
TRAIN_PATH = Path("data/your_train.csv")
TEST_PATH = None                      # or Path("data/your_test.csv")
LABEL_COL = "species"                 # or None if you have no group column
```

4. Run it:

```bash
pip install -r requirements.txt
python run.py
```

5. Look at the terminal printout, then open `outputs/`.

`data/iris.csv` is a sample. You can leave it, or replace it with your file.

PCA is not a classifier. It fits axes on the training table, then (if you set `TEST_PATH`) projects the test table onto those same axes.
