import numpy as np
import pandas as pd

def baseline_always_long(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index, name="position")

def baseline_always_flat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df.index, name="position")

def baseline_random(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    pos = rng.choice([-1, 0, 1], size=len(df))
    return pd.Series(pos, index=df.index, name="position")

def baseline_ema_rsi_rule(df: pd.DataFrame) -> pd.Series:
    """
    Exemple règle simple:
    - Long si ema_20 > ema_50 et rsi_14 < 70
    - Short si ema_20 < ema_50 et rsi_14 > 30
    - sinon Flat
    """
    d = df.copy()
    pos = np.zeros(len(d), dtype=int)

    long_cond = (d["ema_20"] > d["ema_50"]) & (d["rsi_14"] < 70)
    short_cond = (d["ema_20"] < d["ema_50"]) & (d["rsi_14"] > 30)

    pos[long_cond.values] = 1
    pos[short_cond.values] = -1
    return pd.Series(pos, index=d.index, name="position")
