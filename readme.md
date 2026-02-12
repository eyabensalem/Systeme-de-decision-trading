SYSTEM-DE-DECISION-TRADING/
│
├─ README.md
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

# Partie RL : 

# Reinforcement Learning Design (GBPUSD M15)

## 1) Business problem
Goal: learn a trading policy on M15 candles to maximize risk-adjusted performance.
Constraints: transaction costs, no lookahead, strict temporal splits.
Horizon: M15, episode = one year (or fixed window).

## 2) Data
Input: feature-ready M15 dataset (2022/2023/2024 parquet).
Alignment: action at time t is applied to return t->t+1 (shifted position).
Costs: transaction_cost applied when position changes.

## 3) State (observation)
Observation = vector of numeric features at time t (e.g., returns, EMA, RSI, ATR, MACD, ADX, candle features).
Normalization: standardize using train statistics (mean/std).
Warm-up: initial rows removed by feature engineering already (EMA200, etc.).

## 4) Action
Discrete actions A = {-1, 0, +1} corresponding to short/flat/long.

## 5) Reward
Reward(t) = log-return(t+1) * position(t) - transaction_cost * |position(t) - position(t-1)|
Optionally risk penalty: reward -= lambda * (drawdown_increment or volatility).

## 6) Environment
Simulator iterates sequentially over time.
Includes: costs, no slippage (or optional slippage), terminal at end of episode.

## 7) Algorithm choice + justification
Chosen: PPO (stable for discrete actions, works with continuous observation vectors, good default baseline).
Alternative: DQN (works but can be less stable with noisy rewards).

## 8) Key hyperparameters
- gamma: 0.99
- learning_rate: 3e-4
- n_steps: 2048
- batch_size: 64
- ent_coef: 0.0–0.01
- seed: 42

## 9) Evaluation protocol
Strict temporal split:
- Train: 2022
- Validation: 2023 (optional tuning)
- Test: 2024 (final, never used in training)
Metrics:
- Profit cumulé (final equity)
- Max drawdown
- Sharpe
- Profit factor
Stress tests: transaction_cost sensitivity, threshold changes, regime changes.
