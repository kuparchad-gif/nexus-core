# safe_deploy_release.py
# Build a ZIP64 release from /src with excludes + manifest.
import os, sys, json, hashlib, shutil, tempfile, time, uuid
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

EXCLUDE_DIRS = {
    ".git", ".github", ".venv", "__pycache__", ".pytest_cache",
    "deployment_output", "releases", ".continue", ".idea", ".vscode",
    "node_modules", "dist", "build", "tmp", "logs",
    "models", "checkpoints", "weights", "datasets", "data",
    "assets_video", "video", "videos",
}
EXCLUDE_GLOBS = [
    "*.pyc", "*.pyo", "*.log", "*.tmp", "*.bak", "*.DS_Store",
    "*.mp4", "*.mov", "*.mkv", "*.avi", "*.wav", "*.flac", "*.zip",
    "*.tar", "*.tar.gz", "*.7z", "*.zst", "*.bin", "*.onnx",
]

def should_skip(path: Path) -> bool:
    name = path.name.lower()
    if path.is_dir() and name in {d.lower() for d in EXCLUDE_DIRS}:
        return True
    # glob-style excludes
    for pat in EXCLUDE_GLOBS:
        if path.match(pat):
            return True
    # typical cache folders anywhere in tree
    parts = {p.lower() for p in path.parts}
    if "__pycache__" in parts or ".venv" in parts or "node_modules" in parts:
        return True
    return False

def hash_file(p: Path, algo="sha256") -> str:
    h = hashlib.new(algo)
    with p.open("rb", buffering=1024*1024) as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    # Resolve project root
    # script: ...\src\service\core\safe_deploy_release.py → root = two levels up from /src
    script = Path(__file__).resolve()
    src_dir = Path(os.environ.get("PYTHONPATH", "")).resolve() if os.environ.get("PYTHONPATH") else script.parents[3] / "src"
    if not src_dir.exists():
        # fallback: assume standard layout
        src_dir = script.parents[3] / "src"
    root = src_dir.parent
    deploy_dir = root / "deploy" / "updates"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    ver = time.strftime("%Y.%m.%d.%H%M%S", time.localtime())
    zip_path = deploy_dir / f"lillith_release_{ver}.zip"

    staging = Path(tempfile.gettempdir()) / f"lillith_src_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=True)
    staged_src = staging / "src"

    # Copy tree with excludes
    def ig(src, names):
        skipped = set()
        for n in names:
            p = Path(src) / n
            if should_skip(p):
                skipped.add(n)
        return skipped

    print(f"[INFO] Staging from: {src_dir}")
    if not src_dir.exists():
        print(f"[ERR] Source dir not found: {src_dir}")
        return 2
    shutil.copytree(src_dir, staged_src, ignore=ig, dirs_exist_ok=True)

    # Build ZIP64
    print(f"[INFO] Zipping → {zip_path}")
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, allowZip64=True) as zf:
        for p in staged_src.rglob("*"):
            if p.is_dir() or should_skip(p):
                continue
            rel = p.relative_to(staging)
            zf.write(p, rel.as_posix())

    # Hash + manifest
    sha = hash_file(zip_path, "sha256")
    manifest = {
        "version": ver,
        "zip": zip_path.name,
        "sha256": sha,
        "notes": "oneshot build",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    (deploy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Built {zip_path}")
    print(f"[OK] SHA256 {sha}")

    # Cleanup
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
