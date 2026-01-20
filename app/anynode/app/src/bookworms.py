# crawlers/bookworms.py — log & env metadata harvester
import os, sys, json, hashlib, time, glob, argparse, platform
def sha256(p):
    try:
        with open(p,'rb') as f: return hashlib.sha256(f.read()).hexdigest()
    except Exception: return None
def collect_logs(paths):
    out = []
    for pat in paths:
        for p in glob.glob(pat):
            try:
                st = os.stat(p)
                out.append({
                    "path": p,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "sha256": sha256(p)
                })
            except Exception: pass
    return out
def collect_env():
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "env_keys": sorted([k for k in os.environ.keys() if k.upper() in ("PATH","HOME","USER","SHELL")])
    }
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=["/var/log/*.log","*.log"])
    ap.add_argument("--out", default="bookworms_report.json")
    args = ap.parse_args()
    report = {"ts": time.time(), "env": collect_env(), "logs": collect_logs(args.logs)}
    with open(args.out,"w") as f: json.dump(report, f, indent=2)
    print("wrote", args.out)
