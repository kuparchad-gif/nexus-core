
import asyncio, os, json, numpy as np
import httpx

META_BRAIN = os.getenv("META_BRAIN_URL","http://127.0.0.1:8080")

async def main():
    sig = np.random.rand(13).tolist()
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{META_BRAIN}/wire", json={"signal": sig, "phase": 0})
        print("Meta-brain:", r.status_code, r.text[:200])

if __name__ == "__main__":
    asyncio.run(main())
