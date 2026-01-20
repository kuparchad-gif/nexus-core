import os, asyncio, base64, json, nats
from nimcodec import encode as nim_encode, decode as nim_decode
from nim_svc.main import ollama_infer

NATS_URL = os.getenv("NATS_URL", "nats://127.0.0.1:4222")
REQ_SUBJ = os.getenv("NIM_REQ_SUBJ", "nim.infer.req")
RES_SUBJ = os.getenv("NIM_RES_SUBJ", "nim.infer.res")
TILES    = int(os.getenv("NIM_TILES", "64"))

async def main():
    nc = await nats.connect(servers=[NATS_URL])
    async def handler(msg):
        try:
            j = json.loads(msg.data.decode("utf-8"))
            prompt = j.get("prompt","")
            stream = bool(j.get("stream", False))
            if not stream:
                text = await ollama_infer(prompt, stream=False)
                enc = nim_encode(text.encode("utf-8"), data_tiles_per_frame=TILES)
                out = {"encoded_b64": base64.b64encode(enc).decode("ascii"), "tiles": TILES}
                await nc.publish(msg.reply or RES_SUBJ, json.dumps(out).encode())
            else:
                async for piece in ollama_infer(prompt, stream=True):
                    enc = nim_encode(piece.encode("utf-8"), data_tiles_per_frame=TILES)
                    out = {"chunk_b64": base64.b64encode(enc).decode("ascii"), "tiles": TILES}
                    await nc.publish(msg.reply or RES_SUBJ, json.dumps(out).encode())
        except Exception as e:
            await nc.publish(msg.reply or RES_SUBJ, json.dumps({"error": str(e)}).encode())
    await nc.subscribe(REQ_SUBJ, cb=handler)
    print(f"NIM NATS bridge online: sub={REQ_SUBJ} pub={RES_SUBJ} tiles={TILES}")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
