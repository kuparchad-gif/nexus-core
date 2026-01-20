
from __future__ import annotations
import logging, json
from .config import LOG_DIR

def _build_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

loki = _build_logger("Trinity", "trinity.log")
loki_audit = _build_logger("TrinityAudit", "trinity_audit.log")

def log_json(event: str, payload: dict, level: str="info"):
    rec = {"event": event, **payload}
    msg = json.dumps(rec, ensure_ascii=False)
    getattr(loki, level, loki.info)(msg)

def audit(event: str, payload: dict):
    loki_audit.info(json.dumps({"event": event, **payload}, ensure_ascii=False))
