from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json

from .config import load_yaml, ARTIFACTS_DIR
from .data.ingest import fetch_ohlcv, save_local, load_local
from .features.engineer import engineer_features
from .strategy.no_model import rule_combo_signals
from .portfolio.positioning import weights_from_signals
from .backtest.sim import simulate
from .storage.qdrant_store import upsert_snapshots
from .schedule.scheduler import make_scheduler
from .live.trade_loop import run_live_loop
from .model.train import train_and_save

def cmd_ingest(args):
    data = fetch_ohlcv(args.symbols, args.start, args.end, interval="1d")
    if not data:
        raise ValueError("No data fetched for symbols")
    path = save_local(data, tag="default")
    print(f"Saved OHLCV to {path}")

def cmd_backtest(args):
    cfg = load_yaml(args.config)
    symbols = cfg["universe"]["symbols"]
    data = load_local(symbols, tag="default")
    if not data:
        raise ValueError("No data found for symbols; run 'ingest' first")
    feats = engineer_features(data, cfg)
    signals = rule_combo_signals(feats, cfg)
    weights = weights_from_signals(signals, data, cfg)
    res = simulate(weights, data, cfg)
    run_dir = ARTIFACTS_DIR / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(run_dir / "weights.parquet")
    pd.DataFrame(res["series"]).to_parquet(run_dir / "series.parquet")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res["metrics"], f)
    print("Backtest metrics:", res["metrics"])

def cmd_store(args):
    cfg = load_yaml("config/config.yaml")
    symbols = cfg["universe"]["symbols"]
    data = load_local(symbols, tag="default")
    if not data:
        raise ValueError("No data found for symbols; run 'ingest' first")
    feats = engineer_features(data, cfg)
    signals = rule_combo_signals(feats, cfg)
    upsert_snapshots(feats, signals)
    print("Upserted snapshots to Qdrant (local + cloud if configured).")

def cmd_trade(args):
    cfg = load_yaml("config/config.yaml")
    symbols = cfg["universe"]["symbols"]
    data = load_local(symbols, tag="default")
    if not data:
        raise ValueError("No data found for symbols; run 'ingest' first")
    feats = engineer_features(data, cfg)
    signals = rule_combo_signals(feats, cfg)
    weights = weights_from_signals(signals, data, cfg)
    print(weights.tail())

def main():
    ap = argparse.ArgumentParser(description="No-Model Trading Suite CLI")
    sub = ap.add_subparsers(dest="cmd")

    i = sub.add_parser("ingest"); i.add_argument("--symbols", nargs="+", required=True); i.add_argument("--start", required=True); i.add_argument("--end", required=True); i.set_defaults(func=cmd_ingest)
    b = sub.add_parser("backtest"); b.add_argument("--config", default="config/config.yaml"); b.set_defaults(func=cmd_backtest)
    s = sub.add_parser("store"); s.set_defaults(func=cmd_store)
    t = sub.add_parser("trade"); t.add_argument("--paper", action="store_true"); t.add_argument("--rebalance", default="daily"); t.set_defaults(func=cmd_trade)
    l = sub.add_parser("live"); l.add_argument("--poll", type=int, default=60); l.add_argument("--lookback", type=int, default=120); l.set_defaults(func=lambda args: run_live_loop(poll_seconds=args.poll, lookback_minutes=args.lookback))
    tr = sub.add_parser("train"); tr.add_argument("--adapter", default="lightgbm", choices=["lightgbm","nomodel"]); tr.set_defaults(func=lambda args: train_and_save(adapter_name=args.adapter))
    sch = sub.add_parser("schedule"); sch.set_defaults(func=lambda args: make_scheduler().start() or print("Scheduler started (runs in-process)."))

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()