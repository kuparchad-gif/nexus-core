import os
import json
from pathlib import Path

SYMBOLIC_MEMORY_DIR = Path("/memory/logs/visitor_sessions/")
SYMBOLIC_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

async def save_symbolic_memory(category: str, memory_object: dict) -> None:
    category_path = SYMBOLIC_MEMORY_DIR / category
    category_path.mkdir(parents=True, exist_ok=True)

    filename = category_path / f"memory_{memory_object['timestamp'].replace(':', '-')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(memory_object, f, ensure_ascii=False, indent=2)
