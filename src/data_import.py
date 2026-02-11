import pandas as pd

def load_m1_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Vérifier colonnes
    required_cols = ["date", "time", "open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    # Fusion date + time
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convertir prix en numérique
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    return df
