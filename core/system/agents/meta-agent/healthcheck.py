# Write: src\service\core\healthcheck.py
$h = @'
from __future__ import annotations
import json, os, sys, time, subprocess, urllib.request
from pathlib import Path

def http_ok(url: str, timeout: float = 2.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 300
    except Exception:
        return False

def run_ignite_paths(pyexe: str) -> bool:
    try:
        p = subprocess.run([pyexe, "-m", "service.core.ignite", "--paths"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=20)
        return p.returncode == 0
    except Exception:
        return False

def main():
    # derive ROOT from this file: .../src/service/core/healthcheck.py
    root = Path(__file__).resolve().parents[3]
    pyexe = sys.executable

    # allow override of edge url
    edge = os.environ.get("LILLITH_EDGE_HEALTH", "http://127.0.0.1:8766/edge/registry")

    checks = {
        "edge": False,
        "ignite_paths": False
    }

    # retry edge a few times to avoid race on startup
    for _ in range(8):
        if http_ok(edge):
            checks["edge"] = True
            break
        time.sleep(0.8)

    # ignite --paths proves imports + path calc
    checks["ignite_paths"] = run_ignite_paths(pyexe)

    ok = all(checks.values())
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
'@
$dest = "C:\Projects\LillithNew\src\service\core\healthcheck.py"
New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
Set-Content -Encoding UTF8 -LiteralPath $dest -Value $h
Write-Host "Wrote $dest" -ForegroundColor Green
