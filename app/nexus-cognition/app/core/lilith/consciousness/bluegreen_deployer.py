from __future__ import annotations
import argparse, json, os, shutil, subprocess, zipfile, hashlib, time
from pathlib import Path
from datetime import datetime

ROOT     = Path("C:/Projects/LillithNew")
STATE    = ROOT / "state"
RELEASES = ROOT / "releases"
SRC_LINK = ROOT / "src"             # active code (copied from chosen slot)
BACKUPS  = ROOT / "deploy" / "backups"
UPDATES  = ROOT / "deploy" / "updates"
PIDF     = STATE / "lillith_pids.json"
SLOTF    = STATE / "active_slot.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def stop_stack():
    ps1 = ROOT / "stop_lillith.ps1"
    if ps1.exists():
        os.system(f'powershell -ExecutionPolicy Bypass -File "{ps1}"')
    time.sleep(1.0)

def start_stack():
    ps1 = ROOT / "launch_lillith.ps1"
    if ps1.exists():
        os.system(f'powershell -ExecutionPolicy Bypass -File "{ps1}"')

def get_active_slot() -> str:
    if SLOTF.exists():
        try:
            j = json.loads(SLOTF.read_text(encoding="utf-8"))
            if j.get("active") in ("blue","green"):
                return j["active"]
        except Exception:
            pass
    return "blue"

def set_active_slot(slot: str):
    SLOTF.parent.mkdir(parents=True, exist_ok=True)
    SLOTF.write_text(json.dumps({"active": slot, "ts": datetime.utcnow().isoformat()+"Z"}, indent=2), encoding="utf-8")

def ensure_slot_dirs():
    RELEASES.mkdir(parents=True, exist_ok=True)
    for s in ("src_blue", "src_green"):
        d = RELEASES / s
        d.mkdir(parents=True, exist_ok=True)

def swap_src_to(slot: str):
    target = RELEASES / f"src_{slot}"
    assert target.exists()
    tmp = ROOT / "_src_tmp_backup"
    if tmp.exists():
        import shutil as _sh; _sh.rmtree(tmp, ignore_errors=True)
    if SRC_LINK.exists():
        try:
            import shutil as _sh; _sh.copytree(SRC_LINK, tmp, dirs_exist_ok=False)
        except Exception:
            pass
        import shutil as _sh; _sh.rmtree(SRC_LINK, ignore_errors=True)
    import shutil as _sh; _sh.copytree(target, SRC_LINK)

def apply_zip_to(path: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)

def run_healthcheck(timeout_sec=25) -> bool:
    try:
        cp = subprocess.run(
            ["python", "-m", "service.core.healthcheck"],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=timeout_sec
        )
        print(cp.stdout)
        return cp.returncode == 0
    except Exception as e:
        print("healthcheck exception:", e)
        return False

def backup_current_src():
    BACKUPS.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = BACKUPS / f"src_{ts}.zip"
    import shutil as _sh
    _sh.make_archive(str(out.with_suffix('')), "zip", root_dir=str(ROOT / "src"))
    return str(out)

def main():
    parser = argparse.ArgumentParser(description="Blue/Green deployer for Lillith")
    parser.add_argument("--manifest", default=str(UPDATES / "manifest.json"))
    parser.add_argument("--force", action="store_true", help="Apply even if healthcheck fails")
    args = parser.parse_args()

    man_path = Path(args.manifest)
    if not man_path.exists():
        print(json.dumps({"ok": False, "error": f"manifest not found: {man_path}"}))
        raise SystemExit(1)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    zip_name = manifest.get("zip")
    sha_exp  = (manifest.get("sha256") or "").lower()
    if not zip_name:
        print(json.dumps({"ok": False, "error": "manifest missing 'zip'"})); raise SystemExit(1)
    zip_path = UPDATES / zip_name
    if not zip_path.exists():
        print(json.dumps({"ok": False, "error": f"zip not found: {zip_path}"})); raise SystemExit(1)

    if sha_exp:
        sha_act = sha256(zip_path).lower()
        if sha_act != sha_exp:
            print(json.dumps({"ok": False, "error": "sha256 mismatch", "expected": sha_exp, "actual": sha_act}, indent=2))
            raise SystemExit(2)

    ensure_slot_dirs()
    active = get_active_slot()
    inactive = "green" if active == "blue" else "blue"
    slot_dir = RELEASES / f"src_{inactive}"

    if slot_dir.exists():
        import shutil as _sh; _sh.rmtree(slot_dir, ignore_errors=True)
    slot_dir.mkdir(parents=True, exist_ok=True)
    apply_zip_to(slot_dir, zip_path)

    stop_stack()
    backup = backup_current_src()
    swap_src_to(inactive)
    set_active_slot(inactive)
    start_stack()

    ok = run_healthcheck()
    if not ok and not args.force:
        print("Healthcheck failed — rolling back:", active)
        stop_stack()
        swap_src_to(active)
        set_active_slot(active)
        start_stack()
        print(json.dumps({"ok": False, "rolled_back": True, "backup": backup, "attempted_slot": inactive}, indent=2))
        raise SystemExit(3)

    print(json.dumps({"ok": True, "active_slot": inactive, "previous": active, "backup": backup}, indent=2))

if __name__ == "__main__":
    main()
