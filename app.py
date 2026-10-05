"""Browser tester for the PCA module.

Drop a CSV / Excel file, pick columns, fit PCA, look at scores and loadings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pca.io import load_table, numeric_columns
from pca.pca import PCA
from pca.plots import loadings_frame, scatter_figure, scree_figure

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES = {
    "Iris (classic 4-feature table)": DATA_DIR / "iris.csv",
    "Sensors (synthetic plant readings)": DATA_DIR / "sensors.csv",
}

st.set_page_config(page_title="PCA tester", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; max-width: 1200px;}
      h1 {letter-spacing: -0.03em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PCA tester")
st.caption("Put your table in. The module standardizes columns, runs SVD, and returns scores.")

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("CSV, TSV, or Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])
    sample_name = st.selectbox("Or use a bundled file", ["—"] + list(SAMPLES))
    st.header("2. Fit")
    standardize = st.checkbox("Standardize columns", value=True)
    st.caption("Leave this on when columns use different units.")


def _load() -> tuple[pd.DataFrame, str] | tuple[None, None]:
    if uploaded is not None:
        try:
            frame = load_table(uploaded.getvalue(), filename=uploaded.name)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            return None, None
        return frame, uploaded.name
    if sample_name != "—":
        path = SAMPLES[sample_name]
        return load_table(path), path.name
    return None, None


frame, source_name = _load()

if frame is None:
    st.info("Upload a file or pick a bundled sample in the sidebar.")
    left, right = st.columns(2)
    with left:
        st.subheader("What to drop in")
        st.markdown(
            """
            - One row per observation
            - Numeric columns become features
            - One optional label column colors the scatter plot
            - Empty numeric cells drop that row
            """
        )
    with right:
        st.subheader("Also a folder")
        st.markdown(
            """
            Put files in `data/`, then:

            ```bash
            python -m pca data/your_file.csv --label group
            ```
            """
        )
    st.stop()

numeric = numeric_columns(frame)
label_candidates = ["(none)"] + [c for c in frame.columns if c not in numeric] + numeric

if not numeric:
    st.error("This file has no numeric columns. PCA needs numbers.")
    st.dataframe(frame.head(20), use_container_width=True)
    st.stop()

st.success(f"Loaded **{source_name}** — {frame.shape[0]} rows, {frame.shape[1]} columns.")

controls = st.columns([2, 1, 1])
with controls[0]:
    default_features = [c for c in numeric if c.lower() not in {"id", "index"}]
    features = st.multiselect("Numeric features", numeric, default=default_features or numeric)
with controls[1]:
    guessed_label = next((c for c in frame.columns if c.lower() in {"species", "status", "label", "class", "target"}), "(none)")
    label_col = st.selectbox(
        "Color / label column",
        label_candidates,
        index=label_candidates.index(guessed_label) if guessed_label in label_candidates else 0,
    )
with controls[2]:
    max_k = max(1, min(8, len(features) if features else 1, max(1, len(frame) - 1)))
    n_components = st.slider("Components", min_value=1, max_value=max_k, value=min(2, max_k))

if len(features) < 1:
    st.warning("Pick at least one numeric feature.")
    st.stop()

work = frame[features].copy()
if label_col != "(none)":
    work[label_col] = frame[label_col]
before = len(work)
work = work.dropna(subset=features)
dropped = before - len(work)

if len(work) < 2:
    st.error("Need at least 2 complete rows after dropping empty cells.")
    st.stop()

if dropped:
    st.warning(f"Dropped {dropped} row(s) with empty numeric cells.")

try:
    model = PCA(n_components=n_components, standardize=standardize)
    scores = model.fit_transform(work[features].to_numpy())
except Exception as exc:
    st.error(str(exc))
    st.stop()

score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
score_frame = pd.DataFrame(scores, index=work.index, columns=score_cols)
if label_col != "(none)":
    score_frame[label_col] = work[label_col].values

st.subheader("Variance")
var_cols = st.columns(min(4, model.n_components_))
for i, (ratio, cumulative) in enumerate(
    zip(model.explained_variance_ratio_, model.cumulative_variance_ratio())
):
    with var_cols[i % len(var_cols)]:
        st.metric(f"PC{i+1}", f"{ratio:.1%}", f"cumulative {cumulative:.1%}")

plot_left, plot_right = st.columns(2)
with plot_left:
    st.pyplot(scree_figure(model), use_container_width=True)
with plot_right:
    labels = work[label_col] if label_col != "(none)" else None
    st.pyplot(scatter_figure(scores, labels), use_container_width=True)

st.subheader("Loadings")
st.caption("How much each original column contributes to each component.")
st.dataframe(loadings_frame(model, features).style.format("{:.3f}"), use_container_width=True)

st.subheader("Scores")
st.dataframe(score_frame.head(25), use_container_width=True)
st.download_button(
    "Download scores CSV",
    score_frame.to_csv(index=False).encode("utf-8"),
    file_name=f"{Path(source_name).stem}_pca_scores.csv",
    mime="text/csv",
)

with st.expander("Raw table"):
    st.dataframe(frame.head(50), use_container_width=True)
