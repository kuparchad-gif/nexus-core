# connectors/stackoverflow/poller.py
import os, time, httpx
from dotenv import load_dotenv
load_dotenv()
KEY = os.getenv("STACKEXCHANGE_KEY"); TAGS = [t.strip() for t in os.getenv("STACKEXCHANGE_TAGS","python").split(",") if t.strip()]
INGEST_URL = os.getenv("INGEST_URL","http://localhost:8080/ingest")
APPEND_URL = os.getenv("APPEND_URL","http://localhost:8080/append-answer")
SE = "https://api.stackexchange.com/2.3"
seen = set()
def run_once():
    with httpx.Client(timeout = 20) as c:
        params = {"order":"desc","sort":"creation","tagged":";".join(TAGS),"site":"stackoverflow","pagesize":30}
        if KEY: params["key"] = KEY
        qs = c.get(f"{SE}/questions", params = params).json().get("items",[])
        for q in qs:
            qid = f"so:{q['question_id']}"
            if qid in seen: continue
            data = {"platform":"stackoverflow","question_id": qid,"question": q.get("title",""),
                  "url": q.get("link",""),"author_id": str(q.get('owner',{}).get('user_id','')),
                  "author_name": q.get('owner',{}).get('display_name',''),"channel_id": ";".join(TAGS),"timestamp": ""}
            c.post(INGEST_URL, json = data); seen.add(qid)
            if q.get("is_answered") and q.get("accepted_answer_id"):
                aid = q["accepted_answer_id"]
                ans = c.get(f"{SE}/answers/{aid}", params = {"order":"desc","sort":"activity","site":"stackoverflow","filter":"withbody"}).json()
                if ans.get("items"):
                    c.post(f"{APPEND_URL}/{qid}", json = {"answer": ans['items'][0].get("body","")})
if __name__ =  = "__main__":
    while True:
        try: run_once()
        except Exception as e: print("SO poll error:", e)
        time.sleep(120)
