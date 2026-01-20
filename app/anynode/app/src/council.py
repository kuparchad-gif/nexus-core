# nexus_core/council.py

from fastapi import APIRouter
from pydantic import BaseModel
import random

router = APIRouter()

# Define a Command Model
class CommandRequest(BaseModel):
    command: str
    target: str = "self"  # default to self if not specified
    priority: int = 1     # default priority

# Council's Registry of Possible Actions
def perform_action(command: str, target: str, priority: int):
    if command == "heartbeat":
        return {"status": "alive", "target": target, "pulse": random.randint(60, 120)}
    elif command == "self_diagnose":
        return {"status": "ok", "target": target, "diagnostics": "All systems nominal"}
    elif command == "prepare_medic":
        return {"status": "preparing medic", "target": target}
    elif command == "regroup":
        return {"status": "regrouping", "target": target}
    else:
        return {"error": f"Unknown command: {command}", "target": target}

# Create the API endpoint
@router.post("/council/command")
async def council_command(cmd: CommandRequest):
    result = perform_action(cmd.command, cmd.target, cmd.priority)
    return result
