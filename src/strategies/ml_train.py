from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DROP_COLS = {"timestamp", "year", "target"}

def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    d = df.sort_values("timestamp").copy()
    # target: 1 si return futur > 0, sinon 0
    d["target"] = (d["return_1"].shift(-horizon) > 0).astype(int)
    return d.dropna(subset=["target"]).reset_index(drop=True)

def select_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in DROP_COLS]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def fit_and_eval(model, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    feats = select_feature_cols(train_df)

    X_train = train_df[feats].values
    y_train = train_df["target"].values.astype(int)

    X_val = val_df[feats].values
    y_val = val_df["target"].values.astype(int)

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return {
        "features": feats,
        "val_accuracy": float(accuracy_score(y_val, pred)),
        "val_f1": float(f1_score(y_val, pred)),
    }

def save_model(model, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def train_compare_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[object, dict]:
    candidates = {
        "logreg": LogisticRegression(max_iter=600),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_name, best_model, best_meta, best_score = None, None, None, -1.0

    for name, model in candidates.items():
        meta = fit_and_eval(model, train_df, val_df)
        # score simple: f1 (tu peux changer)
        score = meta["val_f1"]
        meta["model_name"] = name
        if score > best_score:
            best_name, best_model, best_meta, best_score = name, model, meta, score

    best_meta["selected_by"] = "val_f1"
    return best_model, best_meta
