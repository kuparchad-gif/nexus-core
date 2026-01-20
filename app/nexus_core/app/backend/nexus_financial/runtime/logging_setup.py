import logging, os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "runtime/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "app.log"

def configure_logging(level: int = logging.INFO):
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=5)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # leave existing handlers intact to avoid behavior changes
