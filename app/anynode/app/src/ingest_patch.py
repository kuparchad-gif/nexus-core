#!/usr/bin/env python3
import sys, os, json, time, argparse, zipfile, pathlib, http.client, mimetypes, uuid
import requests

def post_envelopes(memory_url, envelopes):
    url = memory_url.rstrip('/') + "/ingest/envelopes"
    r = requests.post(url, json={"envelopes": envelopes}, timeout=20)
    r.raise_for_status()
    return r.json()

def post_blob(memory_url, zip_path):
    url = memory_url.rstrip('/') + "/ingest/blob"
    with open(zip_path, "rb") as f:
        files = {"file": (os.path.basename(zip_path), f.read())}
    r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--memory-url", default="http://localhost:8860")
    ap.add_argument("--privacy", default="internal")
    args = ap.parse_args()

    ts_ns = str(int(time.time()*1e9))
    env = {
        "v": 1,
        "ts_ns": ts_ns,
        "service": args.service,
        "level": "info",
        "labels": {
            "tenant": args.tenant,
            "project": args.project,
            "service": args.service,
            "topic": "patch",
            "privacy": args.privacy
        },
        "message": f"[patch] ingest {os.path.basename(args.zip_path)}",
        "source": "patch-kit",
        "topic": "patch"
    }
    print("Posting metadata envelope...")
    print(post_envelopes(args.memory_url, [env]))
    print("Posting patch blob...")
    print(post_blob(args.memory_url, args.zip_path))
    print("Done.")

if __name__ == "__main__":
    main()
