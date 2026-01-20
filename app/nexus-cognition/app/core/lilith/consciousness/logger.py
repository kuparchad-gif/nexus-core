
# src/service/core/logger.py
import json
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

LOG_ROOT = Path(os.environ.get("LOG_DIR", "C:/Projects/LillithNew/logs"))
LOG_ROOT.mkdir(parents=True, exist_ok=True)

def _build_handler(name: str) -> RotatingFileHandler:
    path = LOG_ROOT / f"{name}.log.jsonl"
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    return handler

class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)

def _make_logger(stream_name: str) -> logging.Logger:
    lg = logging.getLogger(f"lillith.{stream_name}")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = _build_handler(stream_name)
        h.setFormatter(JsonLineFormatter())
        lg.addHandler(h)
        lg.propagate = False
    return lg

boot_logger     = _make_logger("boot")         # boot events
guardrail_logger= _make_logger("guardrails")   # decay changes, reinforce/advance
council_logger  = _make_logger("council")      # council decisions
risk_logger     = _make_logger("risk")         # Guardian high-risk flags

def log_json(lg: logging.Logger, msg: str, extra: Optional[Dict[str, Any]] = None):
    if extra is None:
        extra = {}
    lg.info(msg, extra={"extra": extra})
