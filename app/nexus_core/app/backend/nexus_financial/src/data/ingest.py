from __future__ import annotations
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
from ..config import DATA_DIR

def fetch_ohlcv(symbols: List[str], start: str, end: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
    data = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, interval=interval, auto_adjust=True)
        if not df.empty:
            df = df.rename(columns=str.lower)
            df.index = pd.to_datetime(df.index)
            data[sym] = df
    return data

def save_local(data: Dict[str, pd.DataFrame], tag: str = "default"):
    base = DATA_DIR / "ohlcv" / tag
    base.mkdir(parents=True, exist_ok=True)
    for sym, df in data.items():
        df.to_parquet(base / f"{sym}.parquet")
    return base

def load_local(symbols: List[str], tag: str = "default") -> Dict[str, pd.DataFrame]:
    base = DATA_DIR / "ohlcv" / tag
    out = {}
    for sym in symbols:
        p = base / f"{sym}.parquet"
        if p.exists():
            out[sym] = pd.read_parquet(p)
    return out
