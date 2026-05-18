import numpy as np
import pandas as pd

LABEL_CANDIDATES = ["label", "Label", "stress", "stress_level", "target", "class"]

def find_label_column(df_in: pd.DataFrame):
    for c in LABEL_CANDIDATES:
        if c in df_in.columns:
            return c
    return None

def standardize_features_df(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out = out.loc[:, ~out.columns.astype(str).str.startswith("Unnamed:")].copy()

    label_col = find_label_column(out)
    if label_col is None:
        raise ValueError(
            "Khong tim thay cot label trong feature dataset. Can mot trong cac cot: "
            + ", ".join(LABEL_CANDIDATES)
        )
    if label_col != "label":
        out = out.rename(columns={label_col: "label"})

    out = out.dropna(subset=["label"]).copy()
    if out.empty:
        raise ValueError("Feature dataset rong sau khi loai bo dong thieu label.")

    if "source_file" not in out.columns:
        out["source_file"] = "precomputed_features"
    if "window_id" not in out.columns:
        out["window_id"] = np.arange(len(out))

    if out["label"].dtype == "O" or str(out["label"].dtype).startswith("category"):
        out["label"] = out["label"].astype(str).str.strip()
    else:
        out["label"] = pd.to_numeric(out["label"], errors="coerce")

    out = out.dropna(subset=["label"]).copy()
    if out.empty:
        raise ValueError("Khong con mau nao sau khi chuan hoa cot label.")

    return out
