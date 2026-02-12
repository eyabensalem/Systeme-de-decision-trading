import pandas as pd

def aggregate_m15(df_m1: pd.DataFrame) -> pd.DataFrame:
    df = df_m1.copy()

    # supprimer doublons timestamp
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # mettre index temporel
    df = df.sort_values("timestamp").set_index("timestamp")

    # agrégation 15 minutes
    m15 = df.resample("15min").agg(
        open_15m=("open", "first"),
        high_15m=("high", "max"),
        low_15m=("low", "min"),
        close_15m=("close", "last"),
        volume_15m=("volume", "sum"),
    )

    # supprimer bougies incomplètes
    m15 = m15.dropna().reset_index()

    return m15
