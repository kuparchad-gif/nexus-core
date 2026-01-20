import argparse, base64, sys, json, httpx
from nimcodec import encode, decode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["enc","dec","infer"])
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--text", type=str, help="text for enc/infer")
    ap.add_argument("--encoded", type=str, help="base64 for dec")
    ap.add_argument("--url", default="http://127.0.0.1:8300")
    args = ap.parse_args()
    if args.mode == "enc":
        data = (args.text or sys.stdin.read()).encode("utf-8")
        out = encode(data, data_tiles_per_frame=args.tiles)
        print(base64.b64encode(out).decode("ascii"))
    elif args.mode == "dec":
        raw = base64.b64decode(args.encoded or sys.stdin.read())
        out = decode(raw, data_tiles_per_frame=args.tiles)
        print(out.decode("utf-8", errors="ignore"))
    else:
        payload = {"prompt": args.text or sys.stdin.read(), "stream": False, "tiles": args.tiles}
        r = httpx.post(args.url + "/infer", json=payload, timeout=60.0)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
