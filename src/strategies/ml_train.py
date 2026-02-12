from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DROP_COLS = {"timestamp", "year"}  # on ne met pas timestamp en feature

def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    d = df.sort_values("timestamp").copy()
    # target: direction du return_1 futur
    d["target"] = (d["return_1"].shift(-horizon) > 0).astype(int)
    return d.dropna(subset=["target"]).reset_index(drop=True)

def select_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in DROP_COLS and c != "target"]
    # garder seulement numériques
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols

def train_logreg(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[object, dict]:
    X_cols = select_feature_cols(train_df)

    X_train = train_df[X_cols].values
    y_train = train_df["target"].values.astype(int)

    X_val = val_df[X_cols].values
    y_val = val_df["target"].values.astype(int)

    model = LogisticRegression(max_iter=500, n_jobs=None)
    model.fit(X_train, y_train)

    p_val = model.predict(X_val)
    metrics = {
        "val_accuracy": float(accuracy_score(y_val, p_val)),
        "val_f1": float(f1_score(y_val, p_val)),
        "n_features": int(len(X_cols)),
    }
    return model, {"features": X_cols, **metrics}

def save_model(model, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
