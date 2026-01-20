# lilith_engine/modules/spawncore/spawn_bot.py

import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 🔌 Optional: Reflex Core Thinking Integration
# from lilith_engine.modules.signal.reflex_core import ReflexCore
# reflex = ReflexCore(name="lilith")

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# In-memory bot registry
bot_registry = {}

# Bot request schema
class BotRequest(BaseModel):
    bot_type: str
    purpose: str
    priority: int = 1

# Lifecycle options (for future state machine)
BOT_STATES = ["initialized", "active", "archived"]

@router.post("/api/spawn_bot")
def spawn_bot(request: BotRequest):
    bot_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    bot_info = {
        "bot_id": bot_id,
        "bot_type": request.bot_type,
        "purpose": request.purpose,
        "priority": request.priority,
        "spawned_at": timestamp,
        "state": "initialized"
    }

    bot_registry[bot_id] = bot_info

    logging.info(f"[SPAWN] Bot {bot_id} created | Purpose: {request.purpose}")

    # 🧠 Reflex thought placeholder
    # thought = reflex.think(f"Spawning bot for: {request.purpose}")
    # logging.info(f"[ReflexCore 🧠] {thought}")

    return {
        "message": "Bot spawned successfully.",
        "bot_id": bot_id,
        "bot_info": bot_info,
        # "reflex_response": thought  # Enable when Reflex is live
    }

@router.get("/api/spawn_bot/registry")
def get_spawn_registry():
    return {
        "count": len(bot_registry),
        "bots": list(bot_registry.values())
    }
