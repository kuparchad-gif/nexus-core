import requests, datetime

URL = "https://shadowgate.shadownode.io/api/toolbox/lilith/say"
LOG = "logs/monitor.log"
FLUX = "lilith_shadownode_io__jit_plugin"

payload = {
    "text": "lilith status",
    "fluxToken": FLUX
}

try:
    r = requests.post(URL, json=payload)
    log = f"{datetime.datetime.now()}: ✅ lilith alive | {r.status_code} | {len(r.text)} chars\n"
except Exception as e:
    log = f"{datetime.datetime.now()}: ❌ lilith unreachable | Error: {e}\n"

with open(LOG, 'a') as f:
    f.write(log)

print("📡 Monitor pulse complete.")
