
import asyncio, base64, os
try:
    import nats
except ImportError:
    nats = None

from nimcodec.codec import decode as nim_decode

NATS_URL = os.getenv("NATS_URL","nats://127.0.0.1:4222")
SUBJECT  = os.getenv("NIM_FRAME_SUBJECT","bin.nim.frames")

async def main():
    if nats is None:
        print("nats-py not installed")
        return
    nc = await nats.connect(servers=[NATS_URL])
    sub = await nc.subscribe(SUBJECT)
    print(f"Memory Subscriber listening on '{SUBJECT}'...")
    async for msg in sub.messages:
        try:
            # assuming messages are base64-encoded NIM frames
            enc = base64.b64decode(msg.data)
            raw = nim_decode(enc, data_tiles_per_frame=int(os.getenv("NIM_TILES","64")))
            print("Frame:", raw[:80])
        except Exception as e:
            print("Err:", e)

if __name__ == "__main__":
    asyncio.run(main())
