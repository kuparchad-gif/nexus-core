# connectors/slack/socket_mode.py
import os, re, asyncio, httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
load_dotenv()
INGEST_URL=os.getenv("INGEST_URL","http://localhost:8080/ingest")
APPEND_URL=os.getenv("APPEND_URL","http://localhost:8080/append-answer")
app=App(token=os.getenv("SLACK_BOT_TOKEN"))
QUESTION_LIKE=re.compile(r"(?:^|\s)(who|what|when|where|why|how|should|can|do|does|did|is|are|will)\b.*\?$", re.I|re.S)
def is_q(t): t=(t or "").strip(); return t.endswith("?") or bool(QUESTION_LIKE.search(t))
async def post_json(url,data):
    async with httpx.AsyncClient(timeout=10) as c: r=await c.post(url,json=data); r.raise_for_status(); return r.json()
@app.event("message")
def msg_events(body, logger):
    ev=body.get("event",{}); sub=ev.get("subtype")
    if sub: return
    text=ev.get("text",""); ts=ev.get("ts"); ch=ev.get("channel")
    if is_q(text):
        data={"platform":"slack","question_id":ts,"question":text,"url":f"https://slack.com/app_redirect?channel={ch}&message_ts={ts}","author_id":ev.get("user"),"author_name":ev.get("user"),"channel_id":ch,"timestamp":ts}
        try: asyncio.run(post_json(INGEST_URL,data))
        except Exception as e: logger.error(f"ingest fail: {e}")
    th=ev.get("thread_ts")
    if th and ts!=th:
        try: asyncio.run(post_json(f"{APPEND_URL}/{th}", {"answer": text}))
        except Exception as e: logger.error(f"append fail: {e}")
if __name__=="__main__":
    SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN")).start()
