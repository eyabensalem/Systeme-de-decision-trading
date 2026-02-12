from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.baselines import (
    baseline_always_long, baseline_always_flat, baseline_random, baseline_ema_rsi_rule
)
from src.strategies.backtest import backtest_m15
from src.strategies.ml_infer import load_model, predict_proba_up
from src.strategies.ml_train import make_target

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = pd.read_parquet(ROOT / "data" / "processed" / "m15_2024_features.parquet").sort_values("timestamp").reset_index(drop=True)
df_ml = make_target(df, horizon=1)

# baselines
strategies = {
    "always_long": baseline_always_long(df_ml),
    "always_flat": baseline_always_flat(df_ml),
    "random": baseline_random(df_ml, seed=42),
    "ema_rsi_rule": baseline_ema_rsi_rule(df_ml),
}

# ML
model, meta = load_model(ROOT / "models" / "v1")
proba = predict_proba_up(df_ml, model, meta["features"])
pos = np.zeros(len(df_ml), dtype=int)
pos[proba.values > 0.55] = 1
pos[proba.values < 0.45] = -1
strategies["ML"] = pd.Series(pos, index=df_ml.index, name="position")

plt.figure(figsize=(10,4))
for name, pos in strategies.items():
    tmp = df_ml.copy()
    tmp["position"] = pos
    bt = backtest_m15(tmp, transaction_cost=0.00005)
    plt.plot(bt["timestamp"], bt["equity"], label=name)

plt.title("Equity curves (2024) - Baselines vs ML")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS / "equity_2024_baselines_vs_ml.png", dpi=150)
plt.close()

print("Saved:", REPORTS / "equity_2024_baselines_vs_ml.png")
