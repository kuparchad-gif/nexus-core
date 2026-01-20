# connectors/github/poller.py
import os, time, httpx
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN"); REPOS = [r.strip() for r in os.getenv("GITHUB_REPOS","").split(",") if r.strip()]
INGEST_URL = os.getenv("INGEST_URL","http://localhost:8080/ingest")
APPEND_URL = os.getenv("APPEND_URL","http://localhost:8080/append-answer")
seen = set()
def is_q(t): return (t or "").strip().endswith("?")
def headers():
    h = {"Accept":"application/vnd.github+json"};
    if TOKEN: h["Authorization"] = f"token {TOKEN}";
    return h
def run_once():
    with httpx.Client(timeout = 15) as c:
        for repo in REPOS:
            issues = c.get(f"https://api.github.com/repos/{repo}/issues", headers = headers(), params = {"state":"open","per_page":30}).json()
            for it in issues:
                qid = f"github:{repo}:{it.get('id')}"
                if is_q(it.get("title","")):
                    data = {"platform":"github","question_id":qid,"question":f"{it.get('title','')}\n\n{it.get('body','')}","url": it.get("html_url",""),
                          "author_id": str(it.get('user',{}).get('id','')), "author_name": it.get('user',{}).get('login',''),
                          "channel_id": repo, "timestamp": it.get("created_at","")}
                    c.post(INGEST_URL, json = data); seen.add(qid)
                coms_url = it.get("comments_url")
                if coms_url:
                    for cm in c.get(coms_url, headers = headers()).json():
                        c.post(f"{APPEND_URL}/{qid}", json = {"answer": cm.get("body","")})
if __name__ =  = "__main__":
    while True:
        try: run_once()
        except Exception as e: print("poll error:", e)
        time.sleep(60)
