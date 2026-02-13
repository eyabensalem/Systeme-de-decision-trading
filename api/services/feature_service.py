from __future__ import annotations
from pathlib import Path
import pandas as pd

class FeatureService:
    def __init__(self, project_root: Path, parquet_relpath: str = "data/processed/m15_2024_features.parquet"):
        self.root = project_root
        self.parquet_path = self.root / parquet_relpath
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

    def get_latest_row(self) -> pd.Series:
        df = pd.read_parquet(self.parquet_path)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        if len(df) == 0:
            raise ValueError("Parquet is empty.")
        return df.iloc[-1]

    def row_to_feature_dict(self, row: pd.Series, required_features: list[str]) -> dict:
        # on renvoie uniquement les features attendues par le modèle
        return {f: float(row.get(f, 0.0)) for f in required_features}
