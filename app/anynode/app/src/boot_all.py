# src/service/core/boot_all.py
from __future__ import annotations
import os, sys, time, json, subprocess, signal
from pathlib import Path

ROOT  = Path("C:/Projects/LillithNew")
SRC   = ROOT / "src"
LOGS  = ROOT / "logs"
PIDF  = ROOT / "state" / "lillith_pids.json"

PY    = sys.executable  # use system python (no venv assumptions)
ENV   = os.environ.copy()
ENV.setdefault("PYTHONPATH", str(SRC))
ENV.setdefault("LILLITH_STATE_DIR", str(ROOT / "state"))
ENV.setdefault("LILLITH_SNAPSHOT_DIR", str(ROOT / "src" / "state"))  # snapshots live inside src for cloud-extract safety

LOGS.mkdir(parents=True, exist_ok=True)
PIDF.parent.mkdir(parents=True, exist_ok=True)

def _spawn(name: str, args: list[str]):
    log = (LOGS / f"{name}.log").open("ab", buffering=0)
    p = subprocess.Popen(args, stdout=log, stderr=log, env=ENV, cwd=str(ROOT))
    return {"name": name, "pid": p.pid, "args": args, "log": str(log.name)}

def main():
    procs = []

    # 1) Edge AnyNode (message bus)
    procs.append(_spawn("edge_anynode", [PY, "-m", "uvicorn", "service.edge.anynode_service:app", "--host", "127.0.0.1", "--port", "8766"]))

    # 2) Optional services (start if present; if module missing they’ll just log import errors and exit harmlessly)
    optional = [
        ("heart",            [PY, "-m", "service.heart.heart_service"]),
        ("vocal_services",   [PY, "-m", "service.cognikubes.vocal_services.vocal_service"]),
        ("visual_cortex",    [PY, "-m", "service.cognikubes.visual_cortex.visual_cortex_service"]),
        ("trinity_viren",    [PY, "-m", "service.cognikubes.trinity.viren_service"]),
        ("trinity_loki",     [PY, "-m", "service.cognikubes.trinity.loki_service"]),
        # subconscious cluster is library-first; no daemon needed
    ]
    for name, cmd in optional:
        try:
            procs.append(_spawn(name, cmd))
        except Exception:
            # keep going; we’re “up or down”, not fragile
            pass

    # 3) Ignite (she boots last; summary proves life)
    procs.append(_spawn("ignite_summary", [PY, "-m", "service.core.ignite", "--snapshots-in-src", "--summary"]))

    # Persist PIDs so stop script can kill cleanly
    PIDF.write_text(json.dumps(procs, indent=2), encoding="utf-8")

    # Gentle wait to print a tiny status to console (full logs in /logs/*.log)
    time.sleep(2)
    print(json.dumps({
        "status": "UP",
        "pids": {p["name"]: p["pid"] for p in procs},
        "logs_dir": str(LOGS),
        "pidfile": str(PIDF)
    }, indent=2))

if __name__ == "__main__":
    main()
