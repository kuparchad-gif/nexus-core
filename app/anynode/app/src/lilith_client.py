# language_processing/app/alexa_bridge/app/lilith_client.py
# Minimal client to call Lilith via the UCE Router (OpenAI-compatible /v1/chat/completions).

import os
import json
import httpx
import asyncio
from typing import Optional, Tuple

UCE_URL = os.getenv("UCE_ROUTER_URL", "http://localhost:8007/v1/chat/completions")
MODEL_ID = os.getenv("LILITH_MODEL_ID", "qwen-14b")
TIMEOUT = float(os.getenv("UCE_TIMEOUT", "12.0"))
ROOM_MAP_PATH = os.getenv("ROOM_MAP_PATH", "/app/room_map.json")

SYSTEM_SAFE = (
    "You are Lilith, speaking through an Alexa device in HANDS-FREE SAFE MODE. "
    "Be brief (<= 3 sentences), avoid device control, payments, or risky instructions. "
    "Answer conversationally and helpfully. If the request implies action, decline and suggest a safer alternative."
)

SYSTEM_FULL = (
    "You are Lilith, the Aethereal Nexus assistant. "
    "Be concise for voice and helpful. Prefer short answers. "
)

def _load_room_map() -> dict:
    try:
        with open(ROOM_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

async def _async_ping() -> Tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(UCE_URL.rsplit("/", 2)[0] + "/models")
            if r.status_code == 200:
                return True, "router_ok"
            return False, f"router_status_{r.status_code}"
    except Exception as e:
        return False, f"router_err_{type(e).__name__}"

async def ping_lilith() -> Tuple[bool, str]:
    return await _async_ping()

def ask_lilith(query: str, device_id: Optional[str], room_hint: Optional[str], user_hash: str, safe_mode: bool, access_token: Optional[str]) -> str:
    system_prompt = SYSTEM_SAFE if safe_mode else SYSTEM_FULL
    room_map = _load_room_map()
    room = room_hint
    if not room and device_id and device_id in room_map:
        room = room_map.get(device_id)

    meta = {
        "device_id": device_id,
        "room": room,
        "user_hash": user_hash,
        "safe_mode": safe_mode,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[meta]{json.dumps(meta)}[/meta]\n{query}"},
    ]
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.6,
    }
    try:
        r = httpx.post(UCE_URL, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return content or "Lilith is quiet right now."
    except Exception as e:
        return "Lilith is having trouble reaching home base."
