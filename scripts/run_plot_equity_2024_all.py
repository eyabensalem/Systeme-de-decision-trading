from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.baselines import (
    baseline_always_long,
    baseline_always_flat,
    baseline_random,
    baseline_ema_rsi_rule
)
from src.strategies.backtest import backtest_m15
from src.strategies.ml_infer import load_model, predict_proba_up
from src.strategies.ml_train import make_target

from stable_baselines3 import PPO
from src.strategies.rl_env import NormStats, TradingEnv


def main():
    ROOT = Path(__file__).resolve().parent.parent
    REPORTS = ROOT / "reports"
    REPORTS.mkdir(exist_ok=True)

    # -----------------------------
    # Load 2024
    # -----------------------------
    df = pd.read_parquet(
        ROOT / "data" / "processed" / "m15_2024_features.parquet"
    ).sort_values("timestamp").reset_index(drop=True)

    df = make_target(df, horizon=1)

    transaction_cost = 0.00005

    strategies = {}

    # -----------------------------
    # Baselines
    # -----------------------------
    strategies["always_flat"] = baseline_always_flat(df)
    strategies["always_long"] = baseline_always_long(df)
    strategies["random"] = baseline_random(df, seed=42)
    strategies["ema_rsi_rule"] = baseline_ema_rsi_rule(df)

    # -----------------------------
    # ML
    # -----------------------------
    model, meta = load_model(ROOT / "models" / "v1")
    proba = predict_proba_up(df, model, meta["features"])

    ml_pos = np.zeros(len(df), dtype=int)
    ml_pos[proba.values > 0.55] = 1
    ml_pos[proba.values < 0.45] = -1

    strategies["ML"] = pd.Series(ml_pos, index=df.index, name="position")

    # -----------------------------
    # RL (PPO)
    # -----------------------------
    rl_dir = ROOT / "models" / "rl_v1"
    import json
    rl_meta = json.loads((rl_dir / "metadata.json").read_text(encoding="utf-8"))

    feats = rl_meta["features"]
    norm = NormStats(
        mean=np.array(rl_meta["norm_mean"]),
        std=np.array(rl_meta["norm_std"])
    )

    env = TradingEnv(
        df=df,
        feature_cols=feats,
        transaction_cost=float(rl_meta["transaction_cost"]),
        norm=norm,
    )

    rl_model = PPO.load(rl_dir / "ppo_model")

    obs, _ = env.reset()
    rl_positions = []
    done = False

    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        rl_positions.append(info["pos"])

    df_rl = df.iloc[:len(rl_positions)].copy()
    strategies["RL_PPO"] = pd.Series(
        rl_positions,
        index=df_rl.index,
        name="position"
    )

    # -----------------------------
    # Plot Equity Curves (FIX SAFE NORMALIZATION)
    # -----------------------------
    plt.figure(figsize=(11, 4.5))

    for name, pos in strategies.items():

        if name == "RL_PPO":
            tmp = df_rl.copy()
            tc = float(rl_meta["transaction_cost"])
        else:
            tmp = df.copy()
            tc = transaction_cost

        tmp["position"] = pos.values
        bt = backtest_m15(tmp, transaction_cost=tc)

        eq = bt["equity"].astype(float).values

        # 🔥 SAFE NORMALIZATION (avoid NaN issue)
        s = pd.Series(eq).dropna()
        if len(s) == 0:
            continue

        base = float(s.iloc[0])
        eq = eq / base

        plt.plot(bt["timestamp"], eq, label=name)

    plt.title("Equity Curves (2024) — Baselines vs ML vs RL")
    plt.xlabel("Time")
    plt.ylabel("Equity (normalized to 1.0)")
    plt.legend()
    plt.tight_layout()

    out = REPORTS / "equity_2024_baselines_vs_ml_vs_rl.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print("Saved:", out)


if __name__ == "__main__":
    main()
