import os, sys, yaml, argparse, json, shutil
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "models" / "registry.yaml"
DEST = ROOT / "models" / "_hub"

def load_registry():
    with open(REG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dl(hf_id: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=hf_id, local_dir=dest_dir / hf_id.replace("/", "__"), local_dir_use_symlinks=False, resume_download=True)
    return local

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=None, help="Only download a specific capability group")
    args = ap.parse_args()

    reg = load_registry()
    api = HfApi()
    for cap, items in reg.items():
        if args.cap and cap != args.cap:
            continue
        for it in items or []:
            if not it.get("enabled"):
                continue
            hf_id = it["hf_id"]
            print(f"\n==> {cap}: {hf_id}")
            try:
                try:
                    card = api.model_info(hf_id)
                    print(f"   • license: {getattr(card, 'license', 'unknown')}   • downloads: {getattr(card, 'downloads', 'n/a')}")
                except Exception:
                    pass
                local = dl(hf_id, DEST)
                print(f"   • saved to: {local}")
            except Exception as e:
                print(f"   !! failed: {e}")
                continue

if __name__ == "__main__":
    main()
