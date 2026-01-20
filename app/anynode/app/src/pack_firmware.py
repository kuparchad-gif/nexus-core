import argparse, os, json, shutil, tempfile, zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (or path) to pack as firmware")
    ap.add_argument("--out", required=True, help="Output firmware zip path")
    ap.add_argument("--name", default=None, help="Optional firmware name")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "firmware"
    root.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=args.model, local_dir=root / "model", local_dir_use_symlinks=False, resume_download=True)
    manifest = {
        "name": args.name or args.model,
        "source": args.model,
        "layout": {
            "onnx": "model/*.onnx",
            "tokenizer": "model/tokenizer*"
        },
        "runner": "python -m onnxruntime",
        "created_by": "Sovereign CI",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob("*"):
            z.write(p, arcname=str(p.relative_to(root.parent)))
    shutil.rmtree(tmp)
    print(f"✅ Firmware packed: {out}")

if __name__ == "__main__":
    main()
