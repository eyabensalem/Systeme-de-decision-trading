from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Baselines (csv)
base = pd.read_csv(REPORTS / "baselines_2024.csv").set_index("strategy")

# ML finance (json)
ml_fin = json.loads((REPORTS / "ml_2024_finance.json").read_text(encoding="utf-8"))
ml_row = pd.DataFrame([ml_fin], index=["ML"]).rename_axis("strategy")

# RL finance (json)
rl_fin = json.loads((REPORTS / "rl_2024_finance.json").read_text(encoding="utf-8"))
rl_row = pd.DataFrame([rl_fin], index=["RL_PPO"]).rename_axis("strategy")

final = pd.concat([base, ml_row, rl_row], axis=0, sort=False)

out = REPORTS / "final_comparison_2024.csv"
final.to_csv(out)
print("Saved:", out)
