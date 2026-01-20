from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULTS = {"dopamine": 0.55, "circadian_status": "awake"}

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _atomic_write(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def load_affect_state(state_dir: str | Path) -> Dict[str, Any]:
    sdir = Path(os.environ.get("LILLITH_STATE_DIR", str(state_dir)))
    candidates = [sdir / "affect.json", sdir / "reward_state.json", sdir / "boot_snapshot.json"]
    out = {}
    for c in candidates:
        d = _read_json(c)
        if not d:
            continue
        if "dopamine" in d:
            out["dopamine"] = float(d.get("dopamine", DEFAULTS["dopamine"]))
        elif "reward" in d and isinstance(d["reward"], dict):
            out["dopamine"] = float(d["reward"].get("dopamine", DEFAULTS["dopamine"]))
        if "circadian_status" in d:
            out["circadian_status"] = d.get("circadian_status", DEFAULTS["circadian_status"])
        elif "circadian" in d and isinstance(d["circadian"], dict):
            out["circadian_status"] = d["circadian"].get("status", DEFAULTS["circadian_status"])
        if "sleep_until" in d:
            out["sleep_until"] = d.get("sleep_until")
        if "dopamine" in out and "circadian_status" in out:
            break
    return {**DEFAULTS, **out}

def save_affect_state(state_dir: str | Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    sdir = Path(os.environ.get("LILLITH_STATE_DIR", str(state_dir)))
    p = sdir / "affect.json"
    cur = _read_json(p) or {}
    cur.update(updates)
    cur["updated_ts"] = int(time.time())
    _atomic_write(p, cur)
    return cur
