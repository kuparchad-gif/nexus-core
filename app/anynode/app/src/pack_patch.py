#!/usr/bin/env python3
import sys, os, zipfile, pathlib, json, hashlib
from datetime import datetime

def pack_patch(patch_dir: str, out_dir: str):
    patch_dir = pathlib.Path(patch_dir)
    meta = patch_dir / "patch.yaml"
    if not meta.exists():
        raise SystemExit(f"patch.yaml missing in {patch_dir}")
    patch_id = patch_dir.name
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / f"{patch_id}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(patch_dir):
            for name in files:
                full = pathlib.Path(root) / name
                arc = str(full.relative_to(patch_dir))
                z.write(full, arc)
    print(out_zip)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pack_patch.py <patch_dir> [out_dir=dist]")
        sys.exit(1)
    patch_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "dist"
    pack_patch(patch_dir, out_dir)
