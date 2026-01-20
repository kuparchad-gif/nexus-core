# services/api/main.py
import os, typing as T, hmac, hashlib
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import httpx

COL  =  os.getenv("COLLECTION_NAME", "nexus_qna")
client  =  QdrantClient(url = os.getenv("QDRANT_URL","http://qdrant:6333"), api_key = os.getenv("QDRANT_API_KEY") or None)
PROVIDER  =  os.getenv("EMBEDDINGS_PROVIDER","local").lower()
MODEL  =  os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
DIM  =  int(os.getenv("EMBEDDING_DIM","384"))
VERIFIER_URL  =  os.getenv("VERIFIER_URL")

async def embed(texts):
    if PROVIDER == "openai":
        from openai import OpenAI
        client_o  =  OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        resp  =  client_o.embeddings.create(model = "text-embedding-3-large", input = texts)
        return [d.embedding for d in resp.data]
    elif PROVIDER == "voyage":
        import voyageai as vo
        client_v  =  vo.Client(api_key = os.getenv("VOYAGE_API_KEY"))
        return client_v.embed(texts, model = "voyage-3").embeddings
    elif PROVIDER == "cohere":
        import cohere
        co  =  cohere.Client(api_key = os.getenv("COHERE_API_KEY"))
        return co.embed(texts = texts, model = "embed-english-v3.0").embeddings
    else:
        from sentence_transformers import SentenceTransformer
        global _st_model
        try: _st_model
        except NameError: _st_model  =  SentenceTransformer(MODEL)
        return _st_model.encode(texts, normalize_embeddings = True).tolist()

def ensure_collection():
    cols  =  [c.name for c in client.get_collections().collections]
    if COL not in cols:
        client.recreate_collection(collection_name = COL, vectors_config = VectorParams(size = DIM, distance = Distance.COSINE))
ensure_collection()

class QAIngest(BaseModel):
    platform: str  =  Field(..., examples = ["discord","irc","web","slack","github","reddit","stackoverflow","discourse"])
    question_id: str
    question: str
    answer: T.Optional[str]  =  None
    url: T.Optional[str]  =  None
    author_id: T.Optional[str]  =  None
    author_name: T.Optional[str]  =  None
    guild_id: T.Optional[str]  =  None
    channel_id: T.Optional[str]  =  None
    thread_id: T.Optional[str]  =  None
    timestamp: T.Optional[str]  =  None

class QueryIn(BaseModel):
    q: str
    top_k: int  =  6
    filter_platform: T.Optional[str]  =  None

app  =  FastAPI(title = "Nexus Q&A Scraper API")

def point_id_of(question_id: str) -> int:
    return abs(hash(question_id)) % (2**63)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

@app.post("/ingest")
async def ingest(payload: QAIngest):
    text_for_vec  =  payload.question.strip()
    if payload.answer: text_for_vec + =  "\n\nAnswer:\n" + payload.answer.strip()
    vec  =  (await embed([text_for_vec]))[0]
    meta  =  {
        "platform": payload.platform, "question_id": payload.question_id, "question": payload.question,
        "answer": payload.answer or "", "url": payload.url or "", "author_id": payload.author_id or "",
        "author_name": payload.author_name or "", "guild_id": payload.guild_id or "",
        "channel_id": payload.channel_id or "", "thread_id": payload.thread_id or "",
        "timestamp": (payload.timestamp or now_iso()),
        "verified": False, "verification_score": 0.0, "verification_reason": "", "evidence": [],
    }
    client.upsert(collection_name = COL, points = [PointStruct(id = point_id_of(payload.question_id), vector = vec, payload = meta)])
    return {"status":"ok","id": point_id_of(payload.question_id)}

@app.post("/append-answer/{question_id}")
async def append_answer(question_id: str, body: dict):
    answer  =  (body.get("answer") or "").strip()
    if not answer: raise HTTPException(400, "answer required")
    sc  =  client.scroll(collection_name = COL, scroll_filter = Filter(must = [FieldCondition(key = "question_id", match = MatchValue(value = question_id))]), limit = 1)
    if not sc.points: raise HTTPException(404, "question not found")
    p  =  sc.points[0]; question  =  (p.payload or {}).get("question","")

    verified, score, reason, evidence  =  False, 0.0, "", []
    if VERIFIER_URL:
        try:
            async with httpx.AsyncClient(timeout = 60) as c:
                r  =  await c.post(VERIFIER_URL, json = {"question": question, "answer": answer})
                r.raise_for_status(); data  =  r.json()
                verified  =  bool(data.get("verified")); score  =  float(data.get("score",0.0))
                reason  =  data.get("reason",""); evidence  =  data.get("evidence",[])
        except Exception as e:
            reason  =  f"verification_error: {e}"

    vec  =  (await embed([question + "\n\nAnswer:\n" + answer]))[0]
    payload  =  dict(p.payload); payload.update({"answer": answer, "verified": verified, "verification_score": score, "verification_reason": reason, "evidence": evidence})
    client.upsert(collection_name = COL, points = [PointStruct(id = p.id, vector = vec, payload = payload)])
    return {"status":"ok","id": p.id, "verified": verified, "score": score}

@app.post("/query")
async def query(body: QueryIn):
    vec  =  (await embed([body.q]))[0]
    the_filter  =  None
    if body.filter_platform:
        the_filter  =  Filter(must = [FieldCondition(key = "platform", match = MatchValue(value = body.filter_platform))])
    res  =  client.search(collection_name = COL, query_vector = vec, limit = body.top_k, query_filter = the_filter, with_payload = True)
    hits  =  []
    for r in res:
        pay  =  r.payload or {}
        hits.append({
            "score": float(r.score), "question": pay.get("question",""), "answer": pay.get("answer",""),
            "verified": pay.get("verified", False), "verification_score": pay.get("verification_score", 0.0),
            "verification_reason": pay.get("verification_reason",""), "evidence": pay.get("evidence", []),
            "url": pay.get("url",""), "meta": {
                "platform": pay.get("platform",""), "author": pay.get("author_name",""),
                "channel_id": pay.get("channel_id",""), "thread_id": pay.get("thread_id",""),
                "timestamp": pay.get("timestamp",""),
            }
        })
    return {"results": hits}

# GitHub webhook receiver
def _gh_signature_ok(secret: str, body: bytes, signature_header: str) -> bool:
    if not signature_header or not secret: return False
    sha_name, signature  =  signature_header.split(" = ",1)
    mac  =  hmac.new(secret.encode(), msg = body, digestmod = hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), signature)

@app.post("/webhooks/github")
async def github_webhook(request: Request, x_hub_signature_256: str  =  Header(None), x_github_event: str  =  Header(None)):
    secret  =  os.getenv("GITHUB_WEBHOOK_SECRET","")
    raw  =  await request.body()
    if not _gh_signature_ok(secret, raw, x_hub_signature_256 or ""):
        raise HTTPException(401, "bad signature")
    event  =  await request.json()
    ev  =  x_github_event or event.get("event")
    async with httpx.AsyncClient(timeout = 10) as c:
        if ev in ("issues","issue_comment"):
            issue  =  event.get("issue") or {}
            repo  =  event.get("repository",{}).get("full_name","")
            url  =  issue.get("html_url","")
            qid  =  f"github:{repo}:{issue.get('id')}"
            title  =  (issue.get("title") or "").strip()
            body  =  (issue.get("body") or "").strip()
            if title.endswith("?"):
                await c.post(os.getenv("INGEST_URL","http://localhost:8080/ingest"), json = {
                    "platform":"github","question_id": qid,"question": f"{title}\n\n{body}","url": url,
                    "author_id": str(issue.get('user',{}).get('id','')), "author_name": issue.get('user',{}).get('login',''),
                    "channel_id": repo, "timestamp": issue.get("created_at","")
                })
            if ev == "issue_comment":
                comment  =  event.get("comment",{})
                if comment:
                    await c.post(os.getenv("APPEND_URL","http://localhost:8080/append-answer") + f"/{qid}", json = {"answer": comment.get("body","")})
    return {"ok": True}
