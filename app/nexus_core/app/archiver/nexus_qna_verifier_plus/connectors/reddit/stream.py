# connectors/reddit/stream.py
import os, praw, httpx
from dotenv import load_dotenv
load_dotenv()
cid = os.getenv("REDDIT_CLIENT_ID"); csecret = os.getenv("REDDIT_CLIENT_SECRET"); ua = os.getenv("REDDIT_USER_AGENT","NexusQnA/1.0")
subs = [s.strip() for s in os.getenv("REDDIT_SUBREDDITS","askscience").split(",") if s.strip()]
INGEST_URL = os.getenv("INGEST_URL","http://localhost:8080/ingest")
def is_q(t): return (t or "").strip().endswith("?")
def main():
    reddit = praw.Reddit(client_id = cid, client_secret = csecret, user_agent = ua)
    with httpx.Client(timeout = 10) as c:
        for sub in subs:
            for post in reddit.subreddit(sub).stream.submissions(skip_existing = True):
                if is_q(post.title):
                    data = {"platform":"reddit","question_id": f"reddit:{post.id}","question": post.title + "\n\n" + (post.selftext or ""),
                          "url": f"https://reddit.com{post.permalink}","author_id": str(getattr(post.author,'id','')),
                          "author_name": getattr(post.author,'name',''),"channel_id": sub,"timestamp": ""}
                    try: c.post(INGEST_URL, json = data)
                    except Exception as e: print("ingest fail:", e)
if __name__ =  = "__main__": main()
