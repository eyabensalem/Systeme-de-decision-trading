from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

final_path = REPORTS / "final_comparison_2024.csv"
df = pd.read_csv(final_path).set_index("strategy")

best = df["final_equity"].astype(float).idxmax()

# Dossiers modèles (chez toi: V1 en majuscule)
ml_dir = "models/V1"
rl_dir = "models/rl_v1"

payload = {"selected_strategy": best}

if str(best).upper().startswith("RL"):
    payload.update({"type": "rl", "dir": rl_dir})
else:
    payload.update({"type": "ml", "dir": ml_dir})

(MODELS / "active_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Saved:", MODELS / "active_model.json")
print("Selected:", payload)
