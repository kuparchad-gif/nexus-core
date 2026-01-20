# security_validator.py
import modal, bleach
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = modal.App("security-validator")
image = modal.Image.debian_slim().pip_install("fastapi", "uvicorn", "bleach")

class ValidateReq(BaseModel):
    html: str

@app.function(image=image)
@modal.web_server(8000)
def api():
    web = FastAPI()
    @web.post("/validate")
    async def validate(req: ValidateReq):
        clean = bleach.clean(
            req.html,
            tags=["p","b","i","ul","li","a","br"],
            attributes={"a": ["href","title"]},
            strip=True
        )
        return {"sanitized": clean}
    return web