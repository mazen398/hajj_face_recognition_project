"""Load a table from disk or memory and pick numeric columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


def load_table(source: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel. `source` is a path or raw file bytes."""
    suffix = _suffix(source, filename)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No file at {path}")
        if suffix in {".csv", ".txt"}:
            return _read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file type: {suffix or path.name}")

    name = filename or "upload.csv"
    suffix = Path(name).suffix.lower()
    from io import BytesIO

    buf = BytesIO(source)
    if suffix in {".csv", ".txt", ""}:
        return _read_csv(buf)
    if suffix == ".tsv":
        return pd.read_csv(buf, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    raise ValueError(f"Unsupported file type: {suffix}")


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def _read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def _suffix(source, filename: str | None) -> str:
    if filename:
        return Path(filename).suffix.lower()
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    return ""
