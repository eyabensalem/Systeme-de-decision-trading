from pathlib import Path
import numpy as np
import pandas as pd

from src.strategies.ml_train import make_target
from src.strategies.ml_infer import load_model, predict_proba
from src.strategies.backtest import backtest_m15
from src.strategies.metrics import summary_metrics

ROOT = Path(__file__).resolve().parent.parent
p24 = ROOT / "data" / "processed" / "m15_2024_features.parquet"
model, meta = load_model(ROOT / "models" / "v1")

df = pd.read_parquet(p24).sort_values("timestamp").reset_index(drop=True)
df = make_target(df, horizon=1)  # pour aligner target si besoin

proba = predict_proba(df, model, meta["features"])

# règle simple : long si proba>0.55, short si proba<0.45, sinon flat
pos = np.zeros(len(df), dtype=int)
pos[proba.values > 0.55] = 1
pos[proba.values < 0.45] = -1
df["position"] = pos

bt = backtest_m15(df, transaction_cost=0.00005)
met = summary_metrics(bt)

print("ML(2024) metrics:", met)

out = ROOT / "reports" / "ml_2024_metrics.json"
out.parent.mkdir(exist_ok=True)
out.write_text(str(met), encoding="utf-8")
print("Saved:", out)
