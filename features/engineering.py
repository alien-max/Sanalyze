import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    v = df["Volume"]

    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    low14 = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-10)
    df["roc"] = c.pct_change(10) * 100

    hl = h - lo
    hpc = (h - c.shift()).abs()
    lpc = (lo - c.shift()).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_mid"] = sma20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-10)
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"] = obv
    df["volume_ema"] = v.ewm(span=20, adjust=False).mean()
    df["volume_ratio"] = v / (df["volume_ema"] + 1e-10)

    df["daily_return"] = c.pct_change()
    df["log_return"] = np.log(c / c.shift())

    low52 = c.rolling(252).min()
    high52 = c.rolling(252).max()
    df["price_position"] = (c - low52) / (high52 - low52 + 1e-10)

    return df

def add_lag_features(df: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"close_lag_{i}"] = df["Close"].shift(i)
        df[f"return_lag_{i}"] = df["daily_return"].shift(i)

    for w in [5, 10, 20]:
        df[f"return_mean_{w}d"] = df["daily_return"].rolling(w).mean()
        df[f"return_std_{w}d"] = df["daily_return"].rolling(w).std()

    return df

def build_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["target_reg"] = future_close
    df["target_cls"] = (future_close > df["Close"]).astype(int)
    return df

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"Open", "High", "Low", "Close", "Volume", "target_reg", "target_cls"}
    return [c for c in df.columns if c not in exclude]

def prepare_dataset(df: pd.DataFrame, horizon: int = 1, lags: int = 20) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = add_lag_features(df, lags=lags)
    df = build_targets(df, horizon=horizon)
    df.dropna(inplace=True)
    return df