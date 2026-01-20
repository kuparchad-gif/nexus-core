from __future__ import annotations
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import Dict, List
from ..config import QDRANT_LOCAL_URL, QDRANT_CLOUD_URL, QDRANT_API_KEY

COLL = "signals_snapshots"

def _client(url: str, api_key: str | None = None):
    if api_key:
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
    return QdrantClient(url=url, prefer_grpc=False)

def ensure_collection(client: QdrantClient, dim: int):
    try:
        cols = client.get_collections().collections
        names = [c.name for c in cols]
        if COLL not in names:
            client.create_collection(
                collection_name=COLL,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            print(f"Created collection '{COLL}' with dimension {dim}")
    except Exception as e:
        raise RuntimeError(f"Failed to create collection '{COLL}': {e}")

def vectorize_row(row: pd.Series, fields: List[str]) -> List[float]:
    return [float(row.get(f, 0.0)) for f in fields]

def upsert_snapshots(features: Dict[str, pd.DataFrame], signals: Dict[str, pd.Series]):
    # choose fields for vector
    fields = ["sma_diff", "rsi14", "atr", "ret1"]
    # local client
    local = _client(QDRANT_LOCAL_URL)
    # attempt cloud if provided
    cloud = _client(QDRANT_CLOUD_URL, QDRANT_API_KEY) if QDRANT_CLOUD_URL and QDRANT_API_KEY else None

    dim = len(fields)
    ensure_collection(local, dim)
    if cloud:
        ensure_collection(cloud, dim)

    pts_local, pts_cloud = [], []
    pid = 0
    for sym, df in features.items():
        s = signals[sym].reindex(df.index).dropna()
        for ts, val in s.items():
            row = df.loc[ts]
            vec = vectorize_row(row, fields)
            meta = {"symbol": sym, "ts": str(ts), "signal": float(val)}
            pid += 1
            p = PointStruct(id=pid, vector=vec, payload=meta)
            pts_local.append(p)
            pts_cloud.append(p)

    if pts_local:
        local.upsert(collection_name=COLL, points=pts_local)
        print(f"Upserted {len(pts_local)} points to local Qdrant")
    if cloud and pts_cloud:
        cloud.upsert(collection_name=COLL, points=pts_cloud)
        print(f"Upserted {len(pts_cloud)} points to cloud Qdrant")