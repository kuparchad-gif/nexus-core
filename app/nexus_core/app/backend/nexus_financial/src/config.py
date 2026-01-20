import os
from dotenv import load_dotenv
import yaml
from pathlib import Path

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data_cache"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_LOCAL_URL = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
