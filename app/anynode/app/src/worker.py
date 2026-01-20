import os, ssl, json, uuid, asyncio, nats, ray
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

QDRANT = os.getenv("QDRANT_URL","http://127.0.0.1:6333")
LLM_BASE= os.getenv("LLM_BASE_URL","http://127.0.0.1:11434")
COLL   = os.getenv("QDRANT_COLLECTION","docs")

def tls_ctx():
    import ssl
    ctx = ssl.create_default_context(cafile="/certs/ca/ca.crt")
    ctx.load_cert_chain("/certs/client/client.crt","/certs/client/client.key")
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

def ensure_collection(client: QdrantClient, name: str, dim: int):
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

@ray.remote
class Indexer:
    def __init__(self, qdrant_url, llm_base):
        self.q = QdrantClient(url=qdrant_url, timeout=30.0)
        self.emb = OllamaEmbeddings(base_url=llm_base, model="nomic-embed-text")
        v = self.emb.embed_query("warmup")
        ensure_collection(self.q, COLL, len(v))
        self.split = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    def index_url(self, url: str):
        docs = WebBaseLoader(url).load()
        chunks = self.split.split_documents(docs)
        texts = [c.page_content for c in chunks]
        vecs  = self.emb.embed_documents(texts)
        pts   = [PointStruct(id=str(uuid.uuid4()), vector=vecs[i], payload={"text":texts[i], "source":url}) for i in range(len(texts))]
        self.q.upsert(collection_name=COLL, points=pts)
        return {"url": url, "chunks": len(pts)}

async def main():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    idx = Indexer.remote(QDRANT, LLM_BASE)
    nc = await nats.connect(servers=[os.getenv("MCP_NATS_URL","nats://host.containers.internal:4222")],
                            tls=tls_ctx())

    async def ping(msg): await nc.publish(msg.reply, b"pong-from-archiver")
    await nc.subscribe("mcp.ping", cb=ping)

    async def run_index(msg):
        payload = json.loads(msg.data.decode("utf-8"))
        url = payload.get("url","https://example.com")
        jid = payload.get("job_id", str(uuid.uuid4()))
        try:
            res = ray.get(idx.index_url.remote(url))
            if msg.reply:
                await nc.publish(msg.reply, json.dumps(res).encode())
            await nc.publish("logs.worms", json.dumps({"ok":True,"job":jid,"res":res}).encode())
        except Exception as e:
            if msg.reply:
                await nc.publish(msg.reply, json.dumps({"error":str(e)}).encode())
            await nc.publish("logs.worms", json.dumps({"ok":False,"job":jid,"err":str(e)}).encode())

    await nc.subscribe("mcp.archiver.index_url", cb=run_index)
    print("mcp-worker online", flush=True)
    while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
