import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "processed" / "m15_2022_features.parquet"

df = pd.read_parquet(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nDate range:")
print(df["timestamp"].min(), "->", df["timestamp"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nBasic stats:")
print(df.describe())
