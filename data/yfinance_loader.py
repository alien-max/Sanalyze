import yfinance as yf
import pandas as pd
import os
import pickle
import hashlib
from datetime import datetime, timedelta

CACHE_DIR = ".cache"
CACHE_TTL_HOURS = 4

def _cache_path(symbol: str, period: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(f"{symbol}_{period}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def _is_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS)

def load_ohlcv(symbol: str, period: str = "2y") -> pd.DataFrame:
    cache_file = _cache_path(symbol, period)
    if _is_cache_valid(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for symbol '{symbol}'. Check the ticker.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)

    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    return df

def get_symbol_info(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName") or info.get("shortName", symbol),
            "currency": info.get("currency", "USD"),
            "type": info.get("quoteType", "Unknown"),
            "exchange": info.get("exchange", ""),
        }
    except Exception:
        return {"name": symbol, "currency": "USD", "type": "Unknown", "exchange": ""}