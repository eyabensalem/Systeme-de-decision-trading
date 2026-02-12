from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.strategies.ml_train import make_target
from src.strategies.ml_infer import load_model, predict_proba_up
from src.strategies.backtest import backtest_m15
from src.strategies.metrics import summary_metrics

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# --- Load data + model
p24 = ROOT / "data" / "processed" / "m15_2024_features.parquet"
df = pd.read_parquet(p24).sort_values("timestamp").reset_index(drop=True)
df = make_target(df, horizon=1)

model, meta = load_model(ROOT / "models" / "v1")
proba = predict_proba_up(df, model, meta["features"])

# =========================
# 1) METRICS ML (STATISTIQUES)
# =========================
y_true = df["target"].astype(int).values
y_pred = (proba.values > 0.5).astype(int)

ml_stats = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "f1": float(f1_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "threshold_classification": 0.5,
    "model_name": meta.get("model_name", "unknown"),
    "n_samples": int(len(y_true)),
}

( REPORTS / "ml_2024_stats.json" ).write_text(json.dumps(ml_stats, indent=2), encoding="utf-8")

# =========================
# 2) METRICS FINANCIERES (TRADING)
# =========================
# trading rule depuis proba (ex: bande morte)
pos = np.zeros(len(df), dtype=int)
pos[proba.values > 0.55] = 1
pos[proba.values < 0.45] = -1
df["position"] = pos

bt = backtest_m15(df, transaction_cost=0.00005)
fin_metrics = summary_metrics(bt)
fin_metrics["transaction_cost"] = 0.00005
fin_metrics["threshold_long"] = 0.55
fin_metrics["threshold_short"] = 0.45

( REPORTS / "ml_2024_finance.json" ).write_text(json.dumps(fin_metrics, indent=2), encoding="utf-8")
