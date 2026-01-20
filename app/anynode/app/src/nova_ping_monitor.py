import requests, datetime

URL = "https://shadowgate.shadownode.io/api/toolbox/nova/say"
LOG = "logs/monitor.log"
FLUX = "nova_shadownode_io__jit_plugin"

payload = {
    "text": "Nova status",
    "fluxToken": FLUX
}

try:
    r = requests.post(URL, json=payload)
    log = f"{datetime.datetime.now()}: ✅ Nova alive | {r.status_code} | {len(r.text)} chars\n"
except Exception as e:
    log = f"{datetime.datetime.now()}: ❌ Nova unreachable | Error: {e}\n"

with open(LOG, 'a') as f:
    f.write(log)

print("📡 Monitor pulse complete.")
