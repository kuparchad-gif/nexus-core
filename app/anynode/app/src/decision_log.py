from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

# Reuse existing logger setup if available
try:
    from src.service.core.logger import guardrail_logger as logger, log_json
except Exception:
    import logging
    logger = logging.getLogger("Viren")
    def log_json(lg, event, obj): lg.info(f"{event}: {obj}")

def log_decision(kind: str, payload: Dict[str, Any]) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "uuid": str(uuid4()),
        **payload
    }
    log_json(logger, f"viren.{kind}", entry)