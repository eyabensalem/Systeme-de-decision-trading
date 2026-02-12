from pathlib import Path
import pandas as pd
from src.strategies.rl_train import train_ppo

ROOT = Path(__file__).resolve().parent.parent

p22 = ROOT / "data" / "processed" / "m15_2022_features.parquet"
df22 = pd.read_parquet(p22)

out_dir = ROOT / "models" / "rl_v1"
train_ppo(
    train_df=df22,
    transaction_cost=0.00005,
    out_dir=out_dir,
    total_timesteps=50_000,
    seed=42,
)

print("Saved RL model to:", out_dir)
