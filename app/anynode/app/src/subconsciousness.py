from fastapi import APIRouter

router = APIRouter()

@router.post("/run")
def run(payload: dict):
    flow = payload.get("flow", "meditation_watch")
    return {"status":"ok","flow":flow}

