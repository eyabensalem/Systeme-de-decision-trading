from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, meta

def predict_proba(df: pd.DataFrame, model, features: list[str]) -> pd.Series:
    X = df[features].values
    proba_up = model.predict_proba(X)[:, 1]
    return pd.Series(proba_up, index=df.index, name="proba_up")
