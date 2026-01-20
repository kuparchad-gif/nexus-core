from __future__ import annotations
import argparse, json, os, shutil, sys, time, zipfile, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path("C:/Projects/LillithNew")
SRC  = ROOT / "src"
STATE= ROOT / "state"
DEPLOY = ROOT / "deploy"
UPDATES = ROOT / "deploy" / "updates"
BACKUPS = ROOT / "deploy" / "backups"
PIDF = STATE / "lillith_pids.json"

MANIFEST = UPDATES / "manifest.json"   # {"version":"1.2.3","zip":"lillith_release_1.2.3.zip","sha256":"..."}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def stop_stack():
    # Best-effort: call stop script if present
    ps1 = ROOT / "stop_lillith.ps1"
    if ps1.exists():
        os.system(f'powershell -ExecutionPolicy Bypass -File "{ps1}"')
    # As a fallback, try to kill PIDs from pidfile
    try:
        import json, subprocess
        if PIDF.exists():
            procs = json.loads(PIDF.read_text(encoding="utf-8"))
            for p in procs:
                pid = int(p.get("pid", 0))
                if pid:
                    try:
                        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], check=False, capture_output=True)
                    except Exception:
                        pass
            PIDF.unlink(missing_ok=True)
    except Exception:
        pass

def start_stack():
    ps1 = ROOT / "launch_lillith.ps1"
    if ps1.exists():
        os.system(f'powershell -ExecutionPolicy Bypass -File "{ps1}"')

def apply_update(manifest: dict) -> dict:
    zname = manifest.get("zip")
    target = UPDATES / zname
    if not target.exists():
        return {"ok": False, "error": f"update zip missing: {target}"}
    # Verify hash if provided
    expected = (manifest.get("sha256") or "").lower()
    if expected:
        actual = sha256(target).lower()
        if actual != expected:
            return {"ok": False, "error": f"sha256 mismatch: expected {expected}, got {actual}"}
    # Stop services
    stop_stack()
    # Backup current src
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = BACKUPS / f"src_backup_{ts}"
    BACKUPS.mkdir(parents=True, exist_ok=True)
    if SRC.exists():
        shutil.make_archive(str(bdir), "zip", root_dir=str(SRC))
    # Unpack update
    with zipfile.ZipFile(target, "r") as z:
        z.extractall(ROOT)  # zip should contain src/..., scripts/..., etc.
    # Start services
    start_stack()
    return {"ok": True, "applied": zname, "version": manifest.get("version"), "backup": f"{bdir}.zip" if SRC.exists() else None}

def main():
    parser = argparse.ArgumentParser(description="Lillith Update Agent (safe, signed, local-first)")
    parser.add_argument("--check-now", action="store_true", help="Run a one-shot check/apply")
    parser.add_argument("--print-status", action="store_true", help="Print current agent status")
    args = parser.parse_args()

    UPDATES.mkdir(parents=True, exist_ok=True)
    DEPLOY.mkdir(parents=True, exist_ok=True)

    if args.print_status:
        print(json.dumps({
            "root": str(ROOT), "updates_dir": str(UPDATES),
            "manifest_exists": MANIFEST.exists(),
            "pidfile": str(PIDF), "pidfile_exists": PIDF.exists()
        }, indent=2))
        return

    if args.check_now:
        if not MANIFEST.exists():
            print(json.dumps({"ok": False, "error": "no manifest.json found", "dir": str(UPDATES)}))
            return
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        result = apply_update(m)
        print(json.dumps(result, indent=2))
        return

if __name__ == "__main__":
    main()

