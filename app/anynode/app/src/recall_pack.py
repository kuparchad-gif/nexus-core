
# scripts/recall_pack.py
"""
Packs a compact recall bundle with paths, env, and high-level system map.
Usage:
  python scripts/recall_pack.py --out recall/recall_context.json
"""
import argparse, os, json, hashlib
from pathlib import Path

DEFAULT_FILES = [
    "config/cognikubes.json",
    "config/cognikubes.ws.sample.json",
    "seeds/lilith_soul_seed.migrated.json",
    "seeds/trust_phases.json",
    "src/lilith/mesh/cognikube_registry.py",
    "src/lilith/mesh/ws_bus.py",
    "src/lilith/mesh/mcp_adapter.py",
    "src/lilith/spine/ws_spine.py",
    "src/lilith/services/loki_ai.py",
    "src/lilith/services/viren_ai.py",
    "scripts/generate_trust_phases.py",
    "scripts/smoke_trust.py",
    "scripts/wire_src_tree.py"
]

def sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "MISSING"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recall/recall_context.json")
    args = ap.parse_args()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    env = { k: os.environ.get(k) for k in ["JWT_SECRET","AES_KEY","COGNIKUBES_CFG","LILITH_SOUL_SEED"] }
    files = []
    for rel in DEFAULT_FILES:
        p = Path(rel)
        files.append({"path": rel, "exists": p.exists(), "sha256": sha256(p)})

    bundle = {
        "env": env,
        "files": files,
        "map": {
            "ai": ["lilith.spine", "loki.ai", "viren.ai"],
            "transport": ["ws_bus (jwt+aes)", "cognikube_registry (http/udp)", "mcp_adapter"],
            "policy": ["soul_seed", "trust_phases (30-year)"]
        }
    }
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"[ok] wrote {outp}")

if __name__ == "__main__":
    main()
