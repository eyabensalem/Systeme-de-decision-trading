import numpy as np
import pandas as pd

def backtest_m15(
    df: pd.DataFrame,
    position_col: str = "position",
    price_col: str = "close_15m",
    transaction_cost: float = 0.00005
) -> pd.DataFrame:
    """
    Backtest M15: position ∈ {-1,0,1}
    Retourne un DataFrame avec equity + returns nettes.
    """

    d = df.sort_values("timestamp").copy()

    # log returns
    d["ret"] = np.log(d[price_col]).diff()

    # signal/position (pandas series)
    d["pos"] = pd.to_numeric(d[position_col], errors="coerce").fillna(0).clip(-1, 1)

    # stratégie sans lookahead (position t-1 appliquée au return t)
    d["strat_ret_gross"] = d["pos"].shift(1).fillna(0) * d["ret"]

    # coût de transaction
    d["trade"] = d["pos"].diff().abs().fillna(0)  # 0,1,2
    d["cost"] = d["trade"] * transaction_cost

    d["strat_ret_net"] = d["strat_ret_gross"] - d["cost"]

    # equity curve
    d["equity"] = np.exp(d["strat_ret_net"].cumsum())

    return d
