#!/usr/bin/env python3
"""
sync_from_resource.py -- additive or mirror sync with include/exclude patterns
Usage:
  python sync_from_resource.py --src /path/to/good --dst /repo/root [--mirror] [--include "*.py" --include "*.ts"] [--exclude ".git" --exclude "node_modules"]
"""
import argparse, fnmatch, os, shutil, sys, pathlib

def should_skip(path: str, excludes):
    parts = pathlib.Path(path).parts
    for ex in excludes:
        if ex in parts or fnmatch.fnmatch(path, ex):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[".git",".venv","node_modules","dist","build",".qdrant",".pytest_cache",".cache","logs","coverage","tmp","pack",".DS_Store","Thumbs.db"])
    args = ap.parse_args()

    src = pathlib.Path(args.src).resolve()
    dst = pathlib.Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    includes = args.include or ["*"]
    def matches_includes(p):
        rel = str(p.relative_to(src))
        return any(fnmatch.fnmatch(rel, pat) for pat in includes)

    # Mirror mode: delete files in dst that don't exist in src (respect excludes)
    if args.mirror:
        for root, dirs, files in os.walk(dst):
            for name in files:
                dp = pathlib.Path(root)/name
                rel = dp.relative_to(dst)
                sp = src/rel
                if should_skip(str(rel), args.exclude): 
                    continue
                if not sp.exists():
                    dp.unlink()

    # Copy files
    for root, dirs, files in os.walk(src):
        rootp = pathlib.Path(root)
        # prune excluded dirs
        dirs[:] = [d for d in dirs if not should_skip(str(rootp/ d), args.exclude)]
        for f in files:
            sp = rootp / f
            rel = sp.relative_to(src)
            if should_skip(str(rel), args.exclude):
                continue
            if not matches_includes(sp):
                continue
            dp = dst / rel
            dp.parent.mkdir(parents=True, exist_ok=True)
            # Only overwrite if source is newer or missing
            if not dp.exists() or sp.stat().st_mtime > dp.stat().st_mtime:
                shutil.copy2(sp, dp)

if __name__ == "__main__":
    main()
