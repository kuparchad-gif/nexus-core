from __future__ import annotations
import json, os, sys, time, subprocess
from pathlib import Path

ROOT  = Path("C:/Projects/LillithNew")
STATE = ROOT / "state"
PIDF  = STATE / "lillith_pids.json"
LAUNCH = ROOT / "launch_lillith.ps1"

CHECK_EVERY_SEC = 15

def is_running(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in out.stdout
    except Exception:
        return False

def main():
    while True:
        try:
            if not PIDF.exists():
                # try to (re)launch
                os.system(f'powershell -ExecutionPolicy Bypass -File "{LAUNCH}"')
                time.sleep(5)
            else:
                procs = json.loads(PIDF.read_text(encoding="utf-8"))
                any_alive = False
                for p in procs:
                    pid = int(p.get("pid", 0))
                    if pid and is_running(pid):
                        any_alive = True; break
                if not any_alive:
                    os.system(f'powershell -ExecutionPolicy Bypass -File "{LAUNCH}"')
                    time.sleep(5)
        except Exception:
            pass
        time.sleep(CHECK_EVERY_SEC)

if __name__ == "__main__":
    main()
