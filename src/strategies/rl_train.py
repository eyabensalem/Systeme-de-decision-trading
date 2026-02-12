from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from stable_baselines3 import PPO

from src.strategies.ml_train import make_target  # juste pour drop la dernière ligne future si besoin
from src.strategies.rl_env import TradingEnv, compute_norm_stats


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    # on garde uniquement numériques et on exclut timestamp/target/year si présents
    drop = {"timestamp", "target", "year"}
    cols = [c for c in df.columns if c not in drop]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols


def train_ppo(
    train_df: pd.DataFrame,
    transaction_cost: float,
    out_dir: Path,
    total_timesteps: int = 200_000,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # safety: align like ML (avoid last-row target shift issues)
    train_df = make_target(train_df, horizon=1)

    feats = select_feature_cols(train_df)
    norm = compute_norm_stats(train_df, feats)

    env = TradingEnv(
        df=train_df,
        feature_cols=feats,
        transaction_cost=transaction_cost,
        norm=norm,
    )

    policy_kwargs = dict(net_arch=[32, 32])  # au lieu de plus gros

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=32,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=total_timesteps)

    model.save(out_dir / "ppo_model")
    meta = {
        "algo": "PPO",
        "transaction_cost": transaction_cost,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "features": feats,
        "norm_mean": norm.mean.tolist(),
        "norm_std": norm.std.tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
