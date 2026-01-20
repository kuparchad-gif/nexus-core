from __future__ import annotations
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
from ..config import ARTIFACTS_DIR
from ..data.ingest import fetch_ohlcv, save_local
from ..features.engineer import engineer_features
from ..strategy.no_model import rule_combo_signals
from ..portfolio.positioning import weights_from_signals
from ..backtest.sim import simulate
from ..storage.qdrant_store import upsert_snapshots
from ..config import load_yaml

log = logging.getLogger(__name__)

def make_scheduler(cfg_path: str = "config/config.yaml") -> BackgroundScheduler:
    cfg = load_yaml(cfg_path)
    symbols = cfg["universe"]["symbols"]

    sched = BackgroundScheduler(timezone="UTC")

    def nightly_pipeline():
        try:
            # Re-ingest last window and recompute
            data = fetch_ohlcv(symbols, cfg["universe"]["start"], cfg["universe"]["end"], interval="1d")
            save_local(data, tag="default")
            feats = engineer_features(data, cfg)
            sigs = rule_combo_signals(feats, cfg)
            W = weights_from_signals(sigs, data, cfg)
            res = simulate(W, data, cfg)
            from pathlib import Path
            import json, pandas as pd
            run_dir = ARTIFACTS_DIR / "latest"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res["series"]).to_parquet(run_dir / "series.parquet")
            W.to_parquet(run_dir / "weights.parquet")
            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(res["metrics"], f)
            upsert_snapshots(feats, sigs)
            log.info("Nightly pipeline completed.")
        except Exception as e:
            log.exception("Nightly pipeline failed: %s", e)

    # 23:30 UTC weekdays
    sched.add_job(nightly_pipeline, "cron", day_of_week="mon-fri", hour=23, minute=30, id="nightly")
    return sched
