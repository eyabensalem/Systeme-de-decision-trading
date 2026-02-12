# Person 1 – Data Engineering Pipeline

## Source Data
- GBP/USD M1 historical data (2022–2024)
- CSV without header → parsed manually

## Processing Steps
1. Import M1 data and build timestamp
2. Quality checks:
   - gaps > 60s detected (market closures/weekends)
   - duplicates removed
3. Aggregation to M15 OHLCV candles
4. Cleaning:
   - removed invalid prices
   - ensured OHLC consistency
5. Feature Engineering:
   - Returns (log returns)
   - Trend (EMA20, EMA50, EMA200)
   - Momentum (RSI14, MACD)
   - Volatility (ATR, rolling std)
   - Candle structure (body, wicks)
6. Warm‑up removal due to long indicators (EMA200)

## Output Files
Generated datasets:
- m15_2022_features.parquet
- m15_2023_features.parquet
- m15_2024_features.parquet

## EDA
Basic validation plots generated:
- price evolution
- returns distribution
