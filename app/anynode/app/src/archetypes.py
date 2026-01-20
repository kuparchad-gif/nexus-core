# eden_portal/api/archetypes.py
from fastapi import APIRouter
import json
from pathlib import Path

router = APIRouter()

ARCHETYPE_PATH = Path("data/archetypes.json")

@router.get("/api/archetypes")
def get_archetypes():
    with ARCHETYPE_PATH.open() as f:
        return json.load(f)

@router.get("/api/archetypes/{name}")
def get_archetype_by_name(name: str):
    with ARCHETYPE_PATH.open() as f:
        data = json.load(f)
    match = next((a for a in data["archetypes"] if a["name"].lower() == name.lower()), None)
    return match or {"error": "Archetype not found"}
