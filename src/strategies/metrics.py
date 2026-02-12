import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252*24*4) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 5:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))

def profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)

def summary_metrics(bt: pd.DataFrame) -> dict:
    eq = bt["equity"]
    r = bt["strat_ret_net"]
    return {
        "final_equity": float(eq.iloc[-1]),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe_ratio(r),
        "profit_factor": profit_factor(r),
        "n_trades": int((bt["trade"] > 0).sum()),
    }
