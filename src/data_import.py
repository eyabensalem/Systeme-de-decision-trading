import pandas as pd

def load_m1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["date","time","open","high","low","close","volume"]
    )

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y.%m.%d %H:%M",
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    return df
