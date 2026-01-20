import json, os
from fastapi import APIRouter, HTTPException
from .schemas import PromptSpec
from .config import PROMPT_DIR

router  =  APIRouter(prefix = "/v1/prompts", tags = ["prompts"])

def _f(id: str) -> str:
    return os.path.join(PROMPT_DIR, f"{id}.json")

@router.get("")
def list_prompts():
    items  =  []
    for fname in os.listdir(PROMPT_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(PROMPT_DIR, fname), "r", encoding = "utf-8") as f:
                    items.append(json.load(f))
            except Exception:
                continue
    return {"prompts": items}

@router.get("/{id}")
def get_prompt(id: str):
    path  =  _f(id)
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)

@router.post("")
def create_prompt(p: PromptSpec):
    path  =  _f(p.id)
    if os.path.exists(path):
        raise HTTPException(409, "already exists")
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(p.dict(), f, ensure_ascii = False, indent = 2)
    return {"ok": True}

@router.put("/{id}")
def upsert_prompt(id: str, p: PromptSpec):
    if id != p.id:
        raise HTTPException(400, "id mismatch")
    with open(_f(id), "w", encoding = "utf-8") as f:
        json.dump(p.dict(), f, ensure_ascii = False, indent = 2)
    return {"ok": True}

@router.delete("/{id}")
def delete_prompt(id: str):
    path  =  _f(id)
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    os.remove(path)
    return {"ok": True}
