from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.strategies.ml_infer import load_model as load_ml_model, predict_proba_up
from api.services.feature_service import FeatureService

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False


def _existing_dir(root: Path, candidates: list[str]) -> Path:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these model dirs exist: {candidates}")


class InferenceService:
    def __init__(self, project_root: Path):
        self.root = project_root

        active_path = self.root / "models" / "active_model.json"
        if not active_path.exists():
            raise FileNotFoundError("models/active_model.json not found. Run scripts/set_active_model.py")

        self.active = json.loads(active_path.read_text(encoding="utf-8"))
        self.model_type = self.active["type"]

        if self.model_type == "ml":
            self.model_dir = _existing_dir(self.root, ["models/V1", "models/v1"])
            self.model, self.meta = load_ml_model(self.model_dir)
            self.features = self.meta["features"]

        elif self.model_type == "rl":
            if not RL_AVAILABLE:
                raise RuntimeError("RL selected but stable-baselines3 not installed.")

            self.model_dir = _existing_dir(self.root, ["models/rl_v1"])
            self.meta = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.features = self.meta["features"]

            zip_path = self.model_dir / "ppo_model.zip"
            base_path = self.model_dir / "ppo_model"
            self.model = PPO.load(zip_path if zip_path.exists() else base_path)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # -------------------------
    # predict from provided features (debug / legacy)
    # -------------------------
    def predict(self, features_dict: dict) -> tuple[str, float | None]:
        row = {f: float(features_dict.get(f, 0.0)) for f in self.features}
        df = pd.DataFrame([row])

        if self.model_type == "ml":
            proba = float(predict_proba_up(df, self.model, self.features).iloc[0])
            if proba > 0.55:
                return "long", proba
            if proba < 0.45:
                return "short", proba
            return "flat", proba

        # RL (PPO): action 0/1/2
        action, _ = self.model.predict(df.values.astype(np.float32), deterministic=True)
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        else:
            action = int(action)

        if action == 0:
            return "short", None
        if action == 2:
            return "long", None
        return "flat", None

    # -------------------------
    # production mode: build features automatically
    # -------------------------
    def predict_from_features(self, features_dict: dict) -> tuple[str, float | None]:
        return self.predict(features_dict)

    def predict_latest(self) -> dict:
        fs = FeatureService(self.root)  # lit data/processed/m15_2024_features.parquet
        row = fs.get_latest_row()

        features_dict = fs.row_to_feature_dict(row, self.features)
        action, score = self.predict_from_features(features_dict)

        ts = str(row.get("timestamp", "unknown"))
        price = float(row.get("close_15m", 0.0)) if "close_15m" in row.index else None

        return {
            "timestamp": ts,
            "price": price,
            "action": action,
            "score": score,
            "model_type": self.model_type
        }
