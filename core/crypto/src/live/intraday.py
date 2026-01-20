from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import time
import logging

def fetch_intraday(symbol: str, lookback_minutes: int = 60, retries: int = 3) -> pd.DataFrame:
    """Fetch recent 1m bars using yfinance fallback (no key required).
    Returns DataFrame with index as tz-aware UTC timestamps.
    """
    log = logging.getLogger(__name__)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes+5)
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, interval="1m", auto_adjust=True, progress=False)
            if df.empty:
                log.warning(f"No data for {symbol} on attempt {attempt+1}")
                continue
            df = df.rename(columns=str.lower)
            df.index = pd.to_datetime(df.index).tz_convert("UTC")
            return df
        except Exception as e:
            log.error(f"Fetch failed for {symbol}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
    log.error(f"Failed to fetch data for {symbol} after {retries} attempts")
    return pd.DataFrame()