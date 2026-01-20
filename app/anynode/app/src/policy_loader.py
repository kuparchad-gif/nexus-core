from __future__ import annotations
import os, json, sys
from pathlib import Path
from typing import Tuple, Dict, Any

HERE = Path(__file__).resolve()
SRC  = HERE.parents[3]
ROOT = SRC.parent

CFG_SRC   = SRC / "Config"
CFG_ROOT  = ROOT / "Config"
STATE_DIR = Path(os.environ.get("LILLITH_STATE_DIR", str(ROOT / "state")))

def pick_cfg_dir() -> Path:
    src_has  = (CFG_SRC / "guardrails.json").exists() or (CFG_SRC / "sovereignty_policy.json").exists()
    root_has = (CFG_ROOT / "guardrails.json").exists() or (CFG_ROOT / "sovereignty_policy.json").exists()
    return CFG_SRC if src_has else (CFG_ROOT if root_has else CFG_SRC)

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def load_context() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = pick_cfg_dir()
    policy = load_json(cfg / "sovereignty_policy.json")
    guards = load_json(cfg / "guardrails.json")
    state  = load_json(STATE_DIR / "guardrails_state.json")
    return policy, guards, state