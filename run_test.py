from pathlib import Path
from src.data_import import load_m1_csv
from src.m15_agg import aggregate_m15
from src.cleaning import clean_m15
import matplotlib
matplotlib.use("Agg")  # pas de GUI / pas de Tkinter
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
path = ROOT / "data" / "raw" / "DAT_MT_GBPUSD_M1_2022.csv"

df = load_m1_csv(str(path))
m15 = aggregate_m15(df)

m15_clean, rep = clean_m15(m15)

print("Clean report:", rep)
print("M15 clean shape:", m15_clean.shape)
print(m15_clean.head())
from src.features import add_features

feat = add_features(m15_clean)
print("Features shape:", feat.shape)
print(feat.head())

out_feat = ROOT / "data" / "processed" / "m15_2022_features.parquet"
feat.to_parquet(out_feat, index=False)
print("Saved:", out_feat)

from pathlib import Path

ROOT = Path(__file__).resolve().parent
(REPORTS := ROOT / "reports").mkdir(exist_ok=True)

plt.figure(figsize=(10,4))
plt.plot(feat["timestamp"], feat["close_15m"])
plt.title("GBPUSD close (M15)")
plt.tight_layout()
plt.savefig(REPORTS / "price_evolution_2022.png", dpi=150)
plt.close()

plt.figure(figsize=(6,4))
feat["return_1"].hist(bins=100)
plt.title("Return_1 distribution")
plt.tight_layout()
plt.savefig(REPORTS / "returns_hist_2022.png", dpi=150)
plt.close()

print("EDA saved in:", REPORTS)
