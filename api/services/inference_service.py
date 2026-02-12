from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.strategies.ml_infer import load_model as load_ml_model, predict_proba_up


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
            raise FileNotFoundError(
                "models/active_model.json not found. Run scripts/set_active_model.py"
            )

        self.active = json.loads(active_path.read_text(encoding="utf-8"))
        self.model_type = self.active["type"]

        if self.model_type == "ml":
            # robuste à V1 vs v1
            self.model_dir = _existing_dir(self.root, ["models/V1", "models/v1"])
            self.model, self.meta = load_ml_model(self.model_dir)
            self.features = self.meta["features"]

        elif self.model_type == "rl":
            if not RL_AVAILABLE:
                raise RuntimeError("RL selected but stable-baselines3 not installed.")

            self.model_dir = _existing_dir(self.root, ["models/rl_v1"])
            self.meta = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.features = self.meta["features"]

            # robuste à ppo_model.zip
            zip_path = self.model_dir / "ppo_model.zip"
            base_path = self.model_dir / "ppo_model"
            self.model = PPO.load(zip_path if zip_path.exists() else base_path)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, features_dict: dict) -> tuple[str, float | None]:
        # 1 ligne, ordre exact des features du modèle
        row = {f: float(features_dict.get(f, 0.0)) for f in self.features}
        df = pd.DataFrame([row])

        # ----- ML
        if self.model_type == "ml":
            proba = float(predict_proba_up(df, self.model, self.features).iloc[0])
            if proba > 0.55:
                return "long", proba
            if proba < 0.45:
                return "short", proba
            return "flat", proba

        # ----- RL (PPO)
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
