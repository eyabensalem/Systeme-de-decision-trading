from pathlib import Path
from src.data_import import load_m1_csv
from src.m15_agg import aggregate_m15
from src.cleaning import clean_m15

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
