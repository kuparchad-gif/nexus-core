# connectors/discourse/poller.py
import os, time, httpx
from dotenv import load_dotenv
load_dotenv()
BASE = os.getenv("DISCOURSE_BASE_URL","").rstrip("/")
INGEST_URL = os.getenv("INGEST_URL","http://localhost:8080/ingest")
def is_q(t):
    t = (t or "").strip().lower()
    return t.endswith("?") or any(t.startswith(k) for k in ["how","what","why","where","when","should","can","is","are","will"])
def run_once():
    if not BASE: return
    with httpx.Client(timeout = 15) as c:
        latest = c.get(f"{BASE}/latest.json").json()
        for t in latest.get("topic_list",{}).get("topics",[]):
            title = t.get("title",""); tid = t["id"]
            if is_q(title):
                c.post(INGEST_URL, json = {"platform":"discourse","question_id": f"discourse:{tid}","question": title,"url": f"{BASE}/t/{t.get('slug','')}/{tid}","author_id":"", "author_name":"","channel_id": ",".join(t.get("tags",[]) or []),"timestamp": ""})
if __name__ =  = "__main__":
    while True:
        try: run_once()
        except Exception as e: print("discourse poll error:", e)
        time.sleep(120)
