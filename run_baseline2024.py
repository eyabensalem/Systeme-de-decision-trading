from pathlib import Path
import pandas as pd

from src.strategies.baselines import (
    baseline_always_long, baseline_always_flat, baseline_random, baseline_ema_rsi_rule
)
from src.strategies.backtest import backtest_m15
from src.strategies.metrics import summary_metrics

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "m15_2024_features.parquet"
print("Loading:", DATA)

df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)



strategies = {
    "always_long": baseline_always_long(df),
    "always_flat": baseline_always_flat(df),
    "random": baseline_random(df, seed=42),
    "ema_rsi_rule": baseline_ema_rsi_rule(df),
}

rows = []
for name, pos in strategies.items():
    tmp = df.copy()
    tmp["position"] = pos
    bt = backtest_m15(tmp, transaction_cost=0.00005)
    met = summary_metrics(bt)
    met["strategy"] = name
    rows.append(met)

res = pd.DataFrame(rows).set_index("strategy").sort_values("final_equity", ascending=False)
print(res)
out = ROOT / "reports" / "baselines_2024.csv"
out.parent.mkdir(exist_ok=True)
res.to_csv(out)
print("Saved:", out)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.backtest import backtest_m15

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

plt.figure(figsize=(10,4))
for name, pos in strategies.items():
    tmp = df.copy()
    tmp["position"] = pos
    bt = backtest_m15(tmp, transaction_cost=0.00005)
    plt.plot(bt["timestamp"], bt["equity"], label=name)

plt.legend()
plt.title("Equity curves - baselines (2024)")
plt.tight_layout()
plt.savefig(REPORTS / "baselines_2024_equity.png", dpi=150)
plt.close()

print("Saved:", REPORTS / "baselines_2024_equity.png")
