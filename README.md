# PCA

Drop a table in `data/`. Edit three lines in `run.py`. Run it. Read the printout and `outputs/`.

```bash
pip install -r requirements.txt
python run.py
```

In `run.py`:

```python
DATA_PATH = Path("data/your_file.csv")  # csv, tsv, or xlsx
LABEL_COL = "group"                     # or None
N_COMPONENTS = 2
```

`pca.py` is the module. `data/iris.csv` is a sample so you can run it before you add your file.
