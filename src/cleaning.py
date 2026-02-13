import numpy as np
import pandas as pd

def clean_m15(m15: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = m15.sort_values("timestamp").reset_index(drop=True).copy()

    price_cols = ["open_15m","high_15m","low_15m","close_15m"]

    # prix > 0
    before = len(df)
    df = df[(df[price_cols] > 0).all(axis=1)].copy()
    removed_nonpos = before - len(df)

    # cohérence OHLC
    before = len(df)
    df = df[df["high_15m"] >= df["low_15m"]].copy()
    removed_hilo = before - len(df)

    before = len(df)
    df = df[(df["open_15m"].between(df["low_15m"], df["high_15m"])) &
            (df["close_15m"].between(df["low_15m"], df["high_15m"]))].copy()
    removed_openclose = before - len(df)

    # ret log
    df["ret"] = np.log(df["close_15m"]).diff()

    report = {
        "rows_in": int(len(m15)),
        "rows_out": int(len(df)),
        "removed_non_positive": int(removed_nonpos),
        "removed_high_low_incoherent": int(removed_hilo),
        "removed_openclose_outside_range": int(removed_openclose),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
    return df.reset_index(drop=True), report
