# Drop your table here

Put a CSV, TSV, or Excel file in this folder, then:

```bash
python -m pca data/your_file.csv --label your_label_column
```

Or open the Streamlit tester and pick the file from the sidebar.

Rules:

- Each row is one observation.
- Numeric columns become features.
- One optional text/category column can color the scatter plot.
- Empty cells in numeric columns are dropped for that row.

`iris.csv` and `sensors.csv` are bundled so you can run the module before you add your own file.
