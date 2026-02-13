import numpy as np
import pandas as pd

def backtest_m15(
    df: pd.DataFrame,
    position_col: str = "position",
    price_col: str = "close_15m",
    transaction_cost: float = 0.00005,  # 0.5 pip approx (à ajuster)
) -> pd.DataFrame:
    """
    position ∈ {-1, 0, 1}
    PnL basé sur log-returns. Coût appliqué lors des changements de position.
    """
    d = df.sort_values("timestamp").copy()

    if position_col not in d.columns:
        raise ValueError(f"Missing column: {position_col}")
    if price_col not in d.columns:
        raise ValueError(f"Missing column: {price_col}")

    d["pos"] = pd.to_numeric(d[position_col], errors="coerce").fillna(0).clip(-1, 1)

    # returns
    d["ret"] = np.log(d[price_col]).diff()

    # stratégie: position décalée (on agit à t sur ret t->t+1)
    d["strat_ret_gross"] = d["pos"].shift(1).fillna(0) * d["ret"]

    # coûts: chaque fois que la position change
    d["trade"] = d["pos"].diff().abs().fillna(0)  # 0,1,2
    d["cost"] = d["trade"] * transaction_cost

    d["strat_ret_net"] = d["strat_ret_gross"] - d["cost"]

    # equity curve (base 1.0)
    d["equity"] = np.exp(d["strat_ret_net"].cumsum())
    return d
