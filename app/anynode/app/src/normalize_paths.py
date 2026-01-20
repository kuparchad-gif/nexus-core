#!/usr/bin/env python
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (../)
TARGETS = [
    "src/Config",
    "src/configs",
    "src/service",
]

def normalize_value(v: str) -> str:
    # Convert backslashes to forward slashes and strip accidental double slashes
    if isinstance(v, str):
        v = v.replace("\\", "/")
        while "//" in v and not v.startswith("http"):
            v = v.replace("//", "/")
    return v

def normalize_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    changed = False
    def walk(obj):
        nonlocal changed
        if isinstance(obj, dict):
            for k, val in obj.items():
                if isinstance(val, (dict, list)):
                    walk(val)
                elif isinstance(val, str):
                    nv = normalize_value(val)
                    if nv != val:
                        obj[k] = nv
                        changed = True
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                if isinstance(val, (dict, list)):
                    walk(val)
                elif isinstance(val, str):
                    nv = normalize_value(val)
                    if nv != val:
                        obj[i] = nv
                        changed = True
    walk(data)
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed

def main():
    changed_any = False
    for rel in TARGETS:
        base = ROOT / rel
        if not base.exists():
            continue
        for p in base.rglob("*.json"):
            if normalize_file(p):
                print(f"[fixed] {p.relative_to(ROOT)}")
                changed_any = True
    if not changed_any:
        print("[ok] no JSON path fixes needed")

if __name__ == "__main__":
    sys.exit(main())
