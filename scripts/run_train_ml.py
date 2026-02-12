from pathlib import Path
import pandas as pd

from src.strategies.ml_train import make_target, train_compare_models, save_model

ROOT = Path(__file__).resolve().parent.parent
p22 = ROOT / "data" / "processed" / "m15_2022_features.parquet"
p23 = ROOT / "data" / "processed" / "m15_2023_features.parquet"

df22 = pd.read_parquet(p22)
df23 = pd.read_parquet(p23)

df22 = make_target(df22, horizon=1)
df23 = make_target(df23, horizon=1)

model, meta = train_compare_models(df22, df23)

out_dir = ROOT / "models" / "v1"
save_model(model, meta, out_dir)

print("Saved model to:", out_dir)
print("Selected:", meta["model_name"], "| val_f1:", meta["val_f1"], "| val_acc:", meta["val_accuracy"])
print("n_features:", len(meta["features"]))
