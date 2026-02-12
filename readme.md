projet-trading-gbpusd/
│
├─ README.md
├─ README_person1.md
├─ requirements.txt
├─ .gitignore
├─ Dockerfile
│
├─ data/
│  ├─ raw/                      # CSV M1 (IGNORÉ git)
│  └─ processed/                # parquet features 2022/2023/2024 (IGNORÉ git)
│     ├─ m15_2022_features.parquet
│     ├─ m15_2023_features.parquet
│     └─ m15_2024_features.parquet
│
├─ reports/
│  ├─ person1_summary.md         # QC + mini EDA (Personne 1)
│  ├─ price_evolution_2022.png
│  └─ returns_hist_2022.png
│
├─ src/
│  ├─ __init__.py
│  │
│  ├─ data_import.py             # import M1 + timestamp
│  ├─ m15_agg.py                 # M1 → M15 OHLCV
│  ├─ clean_m15.py               # cleaning + checks
│  ├─ features.py                # add_features()
│  │
│  ├─ strategies/
│  │  ├─ __init__.py
│  │  ├─ backtest.py             # moteur backtest + coûts
│  │  ├─ metrics.py              # sharpe, maxDD, profit factor…
│  │  ├─ baselines.py            # baselines (always long, random, règle EMA/RSI…)
│  │  ├─ ml_train.py             # train modèle + save joblib
│  │  ├─ ml_infer.py             # load modèle + predict
│  │  ├─ rl_env.py               # env gym (option)
│  │  └─ rl_train.py             # train RL (option)
│  │
│  └─ evaluation/
│     ├─ __init__.py
│     ├─ eval_pipeline.py        # comparaison finale sur 2024
│     └─ plots.py                # courbes equity/metrics
│
├─ api/
│  ├─ main.py                    # FastAPI app
│  ├─ routers/
│  │  ├─ health.py
│  │  ├─ predict.py              # POST /predict
│  │  └─ model_info.py           # GET /model_version
│  ├─ services/
│  │  ├─ inference_service.py
│  │  └─ feature_service.py
│  └─ schemas/
│     ├─ request.py
│     └─ response.py
│
├─ models/                       # (souvent ignoré si lourd, sinon versionné)
│  └─ v1/
│     ├─ model.joblib
│     └─ metadata.json
│
└─ scripts/
   ├─ run_build_features_all_years.py
   ├─ run_baselines_2024.py
   ├─ run_train_ml.py
   ├─ run_eval_2024.py
   └─ run_eda.py
