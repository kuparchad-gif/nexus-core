# -*- coding: utf-8 -*-
"""
One-shot fixer/booter for Lillith:
- Consumes import_map.json (generated earlier)
- Builds shims for mapped modules (Systems/Utilities → real module)
- Builds no-op stubs for unresolved modules
- Verifies imports
- Boots the first entry: ignite_eden.py / awaken_eden.py / lillith_bootstrap.py
"""

from __future__ import annotations
import json, os, sys, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(r"C:\Projects\LillithNew")
CURRENT = ROOT / "runtime" / "current"
SRC = CURRENT / "src"
MAP_PATH = ROOT / "import_map.json"

ENTRY_CANDIDATES = ("ignite_eden.py", "awaken_eden.py", "lillith_bootstrap.py")

def ensure_pkg_dirs(module_path_rel: str) -> Path:
    """
    Given a module path like 'Systems\\nexus_core\\skill_core.py', ensure
    directories exist under SRC and each level has __init__.py.
    Returns absolute Path to the .py file.
    """
    abs_path = SRC / module_path_rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    p = abs_path.parent
    # add __init__.py up to ...\src
    while True:
        init = p / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        if p == SRC or p == SRC.parent or len(p.parts) < 2:  # safety
            break
        p = p.parent
    return abs_path

def py_identifier_from_tail(tail: str) -> str:
    # Try to convert e.g. 'skill_orchestrator' → 'SkillOrchestrator'
    parts = re.split(r"[^0-9a-zA-Z]+", tail)
    title = "".join(w[:1].upper() + w[1:] for w in parts if w)
    return title or "Stub"

def write_stub(module_rel: str, class_names: List[str] | None = None) -> None:
    """
    Write a minimal no-op stub with given class names (or a best-guess single class).
    """
    mod_file = ensure_pkg_dirs(module_rel)
    if class_names is None or not class_names:
        # Guess a class from filename (without .py)
        guess = py_identifier_from_tail(mod_file.stem)
        class_names = [guess]
    lines = []
    for cls in class_names:
        lines += [
            f"class {cls}:",
            "    def __init__(self, *args, **kwargs):",
            "        pass",
            "    def start(self, *args, **kwargs):",
            "        return True",
            "",
        ]
    mod_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[STUB] {module_rel} ({', '.join(class_names)})")

def write_shim(expected_mod: str, real_mod: str) -> None:
    """
    `expected_mod` is dotted (e.g. Systems.engine.api.manifest_server)
    Create a .py that re-exports from `real_mod`.
    """
    # Convert expected dotted to relative path
    rel = expected_mod.replace(".", "\\") + ".py"
    mod_file = ensure_pkg_dirs(rel)
    shim = (
        '"""Auto-generated shim for legacy import."""\n'
        "import importlib as _imp\n"
        f"_m = _imp.import_module({real_mod!r})\n"
        "globals().update({k: v for k, v in _m.__dict__.items() if not k.startswith('_')})\n"
    )
    mod_file.write_text(shim, encoding="utf-8")
    print(f"[SHIM] {expected_mod}  ->  {real_mod}")

def load_map() -> Dict[str, Any]:
    if not MAP_PATH.exists():
        raise SystemExit(f"import_map.json not found at {MAP_PATH}")
    return json.loads(MAP_PATH.read_text(encoding='utf-8'))

def main() -> int:
    os.environ["PYTHONPATH"] = str(SRC)  # ensure our release tree is importable
    m = load_map()
    decisions: Dict[str, Dict[str, Any]] = m.get("decisions", {})

    # 1) Create shims for mapped items
    for expected, info in decisions.items():
        if info.get("status") == "mapped" and info.get("real"):
            write_shim(expected, info["real"])

    # 2) Create stubs for unresolved Systems.* and Utilities.* items
    unresolved = [k for k, v in decisions.items() if v.get("status") == "unresolved"]
    for name in unresolved:
        # Convert dotted to path and pick a class name
        rel = name.replace(".", "\\") + ".py"
        tail = name.split(".")[-1]
        cls = py_identifier_from_tail(tail)
        write_stub(rel, [cls])

    # 3) Known must-have Utilities (ensure they exist if not already)
    must_utils = {
        "Utilities.secret_loader": ["SecretLoader"],
        "Utilities.deployment_logger": ["DeploymentLogger"],
        "Utilities.memory_manifestor": ["MemoryManifestor"],
        "Utilities.memory_synchronizer": ["MemorySynchronizer"],
    }
    for dotted, classes in must_utils.items():
        rel = dotted.replace(".", "\\") + ".py"
        absf = SRC / rel
        if not absf.exists():
            # Minimal stubs if missing; your earlier run likely created real ones.
            write_stub(rel, classes)

    # 4) Import smoke test (the critical ones)
    critical = [
        "Utilities.secret_loader",
        "Utilities.deployment_logger",
        "Utilities.memory_manifestor",
        "Utilities.memory_synchronizer",
        "Systems.engine.api.manifest_server",
        "Systems.nexus_core.genesis_core",
    ]
    print("\n[SMOKE] importing critical modules...")
    failures = []
    for mod in critical:
        try:
            __import__(mod)
            print("  OK  ", mod)
        except Exception as e:
            print("  FAIL", mod, "->", e)
            failures.append((mod, str(e)))

    if failures:
        print("\n[SMOKE] Some imports still failing. Not booting.")
        for mod, err in failures:
            print("  -", mod, ":", err)
        return 1

    # 5) Find entry and boot
    entry = None
    for pat in ENTRY_CANDIDATES:
        found = list(SRC.rglob(pat))
        if found:
            entry = found[0]
            break
    if not entry:
        print("No entrypoint found under", SRC)
        return 2

    print(f"\n[BOOT] Launching {entry}")
    # chain-load the entry
    os.execv(sys.executable, [sys.executable, str(entry)])

if __name__ == "__main__":
    sys.exit(main())
