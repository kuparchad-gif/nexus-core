# scripts/wire_src_tree.py (pre-crossing stable state - low risk)
import argparse
from pathlib import Path
import shutil
import time

CODE_EXT = {".py", ".ts", ".js"}
DATA_EXT = {".json", ".yaml", ".yml", ".toml", ".ini"}
EXCLUDE_DIRS = {".git", "node_modules", "__pycache__"}

def copy_file(src: Path, dst: Path, force: bool = False, dry_run: bool = False) -> str:
    if dry_run:
        action = "would_copy" if not dst.exists() else "would_overwrite_data" if src.suffix.lower() in DATA_EXT else "would_overwrite_code" if force and src.suffix.lower() in CODE_EXT else "would_skip"
        return f"{action}:{dst}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return f"copied:{dst}"
    if src.suffix.lower() in DATA_EXT:
        shutil.copy2(src, dst)
        return f"overwrote_data:{dst}"
    if src.suffix.lower() in CODE_EXT:
        if force:
            bak = dst.with_suffix(f"{dst.suffix}.bak.{int(time.time())}")
            shutil.copy2(dst, bak)
            shutil.copy2(src, dst)
            return f"overwrote_code:{dst} (backup:{bak.name})"
        return f"skipped_code_exists:{dst}"
    shutil.copy2(src, dst)
    return f"overwrote:{dst}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    if not src.exists():
        raise SystemExit(f"src not found: {src}")

    actions = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        r = Path(root)
        for f in files:
            if f.startswith("."):
                continue
            ext = Path(f).suffix.lower()
            if ext not in CODE_EXT and ext not in DATA_EXT:
                continue
            rel = r.relative_to(src) / f
            actions.append(copy_file(r / f, dest / rel, force=args.force, dry_run=args.dry_run))

    print("\n".join(actions))

if __name__ == "__main__":
    main()