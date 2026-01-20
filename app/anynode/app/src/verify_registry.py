import os, yaml, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "_hub"

def sha256sum(path: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    for base, _, files in os.walk(DEST):
        for fn in files:
            p = Path(base) / fn
            if p.stat().st_size < 1_000_000:
                continue
            print(f"{p}  {p.stat().st_size} bytes  sha256={sha256sum(p)}")

if __name__ == "__main__":
    main()
