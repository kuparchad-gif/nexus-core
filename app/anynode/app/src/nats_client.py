# cognikubes/nexus_core/services/browser_receiver/app/nats_client.py
import asyncio, json, os
from typing import AsyncIterator
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg

_nc = None
_lock = asyncio.Lock()

async def get_nc() -> NATS:
    global _nc
    if _nc and _nc.is_connected:
        return _nc
    async with _lock:
        if _nc and _nc.is_connected:
            return _nc
        _nc = NATS()
        url = os.getenv("NATS_URL", "nats://localhost:4222")
        await _nc.connect(servers=[url], reconnect=True, max_reconnect_attempts=-1)
        return _nc

async def publish(subject: str, payload: dict):
    nc = await get_nc()
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await nc.publish(subject, data)

async def subscribe_stream(pattern: str) -> AsyncIterator[Msg]:
    nc = await get_nc()
    sub = await nc.subscribe(pattern)
    try:
        while True:
            msg = await sub.next_msg()  # no timeout
            yield msg
    finally:
        await nc.unsubscribe(sub.sid)

