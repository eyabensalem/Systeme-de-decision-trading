from pathlib import Path
import json
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from src.strategies.rl_env import NormStats, TradingEnv
from src.strategies.ml_train import make_target
from src.strategies.backtest import backtest_m15
from src.strategies.metrics import summary_metrics

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

model_dir = ROOT / "models" / "rl_v1"
meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

# load 2024
df = pd.read_parquet(ROOT / "data" / "processed" / "m15_2024_features.parquet").sort_values("timestamp").reset_index(drop=True)
df = make_target(df, horizon=1)

feats = meta["features"]
norm = NormStats(mean=np.array(meta["norm_mean"]), std=np.array(meta["norm_std"]))

env = TradingEnv(df=df, feature_cols=feats, transaction_cost=float(meta["transaction_cost"]), norm=norm)
model = PPO.load(model_dir / "ppo_model")

obs, _ = env.reset()
positions = []
timestamps = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(int(action))
    positions.append(info["pos"])
    timestamps.append(df.loc[env.t, "timestamp"])

# align positions length with df (simple)
df_bt = df.iloc[: len(positions)].copy()
df_bt["position"] = positions

bt = backtest_m15(df_bt, transaction_cost=float(meta["transaction_cost"]))
fin = summary_metrics(bt)
fin["model"] = "PPO"
fin["transaction_cost"] = float(meta["transaction_cost"])

(REPORTS / "rl_2024_finance.json").write_text(json.dumps(fin, indent=2), encoding="utf-8")
print("Saved:", REPORTS / "rl_2024_finance.json")
