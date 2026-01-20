import asyncio
import httpx

async def send_heartbeat(target_url: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(target_url + "/pulse/heartbeat")
            if response.status_code == 200:
                return True
            return False
        except Exception:
            return False
