import argparse, json, os, pathlib
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="LilithFirmware")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    src = pathlib.Path(args.model_dir)
    dst = pathlib.Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for p in src.iterdir():
        if p.is_file():
            (dst / p.name).write_bytes(p.read_bytes())

    manifest = {
        "name": args.name,
        "version": args.version,
        "built_at": datetime.utcnow().isoformat() + "Z",
        "source": str(src),
        "artifacts": sorted([x.name for x in dst.iterdir() if x.is_file()]),
        "mpo_metatron": { "enabled": True, "notes": args.notes }
    }
    (dst / "lilith_firmware.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Firmware bundle ready at {dst}")

if __name__ == "__main__":
    main()
