import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("timestamp").copy()

    close = d["close_15m"]
    high  = d["high_15m"]
    low   = d["low_15m"]
    open_ = d["open_15m"]

    d["return_1"] = np.log(close).diff(1)
    d["return_4"] = np.log(close).diff(4)

    d["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    d["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    d["ema_diff"] = d["ema_20"] - d["ema_50"]

    d["rsi_14"] = RSIIndicator(close, window=14).rsi()

    d["rolling_std_20"] = d["return_1"].rolling(20).std()
    d["range_15m"] = high - low

    d["body"] = (close - open_).abs()
    d["upper_wick"] = high - np.maximum(open_, close)
    d["lower_wick"] = np.minimum(open_, close) - low

    d["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
    d["distance_to_ema200"] = (close - d["ema_200"]) / (d["ema_200"] + 1e-9)
    d["slope_ema50"] = d["ema_50"].diff(5)

    d["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    d["rolling_std_100"] = d["return_1"].rolling(100).std()
    d["volatility_ratio"] = d["rolling_std_20"] / (d["rolling_std_100"] + 1e-9)

    d["adx_14"] = ADXIndicator(high, low, close, window=14).adx()
    macd = MACD(close)
    d["macd"] = macd.macd()
    d["macd_signal"] = macd.macd_signal()

    # warm-up (ema200 etc.)
    d = d.dropna().reset_index(drop=True)
    return d
