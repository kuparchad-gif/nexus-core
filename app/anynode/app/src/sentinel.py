# crawlers/sentinel.py — detect unencrypted attempts & kill comms (lock-aware)
import sys, json, subprocess, argparse, os, time

LOCK_PATH = os.environ.get("DRAWBRIDGE_LOCK", "/run/drawbridge.lock")

ACTIONS = {
    "cleartext_blocked": "shutdown",
    "unencrypted_or_bad_aead": "shutdown",
    "policy_block": "alert"
}

def lock_active():
    try:
        with open(LOCK_PATH, "r") as f:
            obj = json.load(f)
        return obj.get("until", 0) > time.time()
    except Exception:
        return False

def act(action):
    if lock_active():
        print("[SENTINEL] Override lock active; suppressing action:", action)
        return
    if action == "shutdown":
        try:
            subprocess.call(["/sbin/ip","link","set","eth0","down"])
        except Exception:
            pass
        print("[SENTINEL] COMMUNICATIONS SHUT DOWN")
    elif action == "alert":
        print("[SENTINEL] ALERT: policy violation")
    else:
        print("[SENTINEL] NOOP")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdin", action="store_true", help="read events from stdin")
    args = ap.parse_args()
    if not args.stdin:
        print("Pipe Nexus Guardian Proxy logs into this script with --stdin"); sys.exit(1)
    for line in sys.stdin:
        try:
            ev = json.loads(line)
            kind = ev.get("event","")
            action = ACTIONS.get(kind)
            if action: act(action)
        except Exception:
            continue
