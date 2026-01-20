from __future__ import annotations
import time
from datetime import datetime, timezone
from .live.intraday import fetch_intraday
from .config import load_yaml
from .features.engineer import engineer_features
from .strategy.no_model import rule_combo_signals
from .portfolio.positioning import weights_from_signals

def run_live_loop(poll_seconds: int = 60, lookback_minutes: int = 120):
    cfg = load_yaml("config/config.yaml")
    symbols = cfg["universe"]["symbols"]
    print("Starting live loop. Ctrl+C to stop.")
    while True:
        # fetch last N minutes for each symbol (very simplified demo)
        data = {}
        for sym in symbols:
            df = fetch_intraday(sym, lookback_minutes=lookback_minutes)
            if not df.empty:
                data[sym] = df.rename(columns=str.lower)
        if data:
            feats = engineer_features(data, cfg)
            sigs = rule_combo_signals(feats, cfg)
            W = weights_from_signals(sigs, data, cfg)
            print(f"[{datetime.now(timezone.utc).isoformat()}] Latest target weights:")
            print(W.tail(1).T)
        else:
            print("No intraday data fetched.")
        time.sleep(poll_seconds)
