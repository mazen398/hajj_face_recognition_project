# PCA lab

Turn a table into principal components, then look at the scores.

The Windows notebook path was not uploaded into this workspace, so this folder is the runnable version: a small PCA module, a `data/` drop zone, a notebook, and a browser tester.

## What is in here

```
pca/                 SVD PCA module
data/                drop your CSV / Excel here
notebooks/pca.ipynb  same module, cell by cell
app.py               browser tester
tests/               checks on iris + reconstruction
```

## Insert your data

1. Copy a CSV, TSV, or Excel file into `data/`.
2. Rows = observations. Numeric columns = features. One optional text column can color the plot.
3. Run one of the three testers below.

Bundled files you can use first:

- `data/iris.csv` — 150 flowers, 4 measurements, `species` label
- `data/sensors.csv` — synthetic plant readings, `status` label

## Run the browser tester

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.port 8765
```

Upload a file in the sidebar, or pick a bundled sample. The page shows explained variance, a scree plot, a PC1–PC2 scatter, loadings, and a scores download.

## Run from the terminal

```bash
python -m pca data/iris.csv --label species --components 2
python -m pca data/your_file.csv --label group --out data/your_file_scores.csv
```

## Run the notebook

```bash
pip install jupyter
jupyter notebook notebooks/pca.ipynb
```

The first code cell adds the repo root to `sys.path` so `from pca import PCA` works.

## Tests

```bash
pytest -q
```
