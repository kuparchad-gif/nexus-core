from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..features.engineer import engineer_features
from ..data.ingest import load_local
from ..config import load_yaml, ARTIFACTS_DIR
from .adapter import NoModelAdapter, LightGBMAdapter
from pathlib import Path
import json

def make_xy(features: Dict[str, pd.DataFrame], horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    frames = []
    ys = []
    for sym, x in features.items():
        # next h-day return label
        y = x["close"].pct_change(horizon).shift(-horizon)
        feats = x.drop(columns=["open","high","low","close","adj close","volume"], errors="ignore").copy()
        # align
        both = feats.join(y.rename("y")).dropna()
        frames.append(both.drop(columns=["y"]))
        ys.append(both["y"])
    X = pd.concat(frames).fillna(0.0)
    y = pd.concat(ys).fillna(0.0)
    return X, y

def train_and_save(config_path: str = "config/config.yaml", adapter_name: str = "lightgbm"):
    cfg = load_yaml(config_path)
    symbols = cfg["universe"]["symbols"]
    data = load_local(symbols, tag="default")
    feats = engineer_features(data, cfg)
    X, y = make_xy(feats, horizon=5)

    if adapter_name == "lightgbm":
        model = LightGBMAdapter()
    elif adapter_name == "nomodel":
        model = NoModelAdapter()
    else:
        raise ValueError("Unknown adapter")

    model.fit(X, y)
    out_dir = ARTIFACTS_DIR / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    if adapter_name == "lightgbm":
        model.save(str(out_dir / "lgb.txt"))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"adapter": adapter_name, "features": list(X.columns)}, f)
    print("Model trained and saved to", out_dir)
