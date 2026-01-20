
# python -m tests.narrative_smoke
import requests, time

BASE = "http://localhost:8030"

def main():
    s = requests.get(f"{BASE}/narrative/status").json()
    print("status:", s)

    sim = requests.post(f"{BASE}/narrative/simulate", json={"seed":"AI cannot be trusted", "neg":0.3, "platforms":3}).json()
    print("simulate:", sim)
    ev = sim["event_id"]

    time.sleep(1)
    pub = requests.post(f"{BASE}/narrative/approve", json={"event_id": ev, "wave": 1, "platforms":["youtube","x"], "approver":"Owner"}).json()
    print("publish wave1:", pub)

if __name__ == "__main__":
    main()
