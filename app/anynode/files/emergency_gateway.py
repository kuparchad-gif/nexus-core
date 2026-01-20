from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

emergency_router  =  APIRouter()

# Dummy user database (replace with secure vault-based lookup later)
USERS  =  {
    "admin": "secretpassword",
    "chad": "edenmasterkey"
}

@emergency_router.post("/eden_console/authenticate")
async def authenticate(request: Request):
    data  =  await request.json()
    username  =  data.get("username")
    password  =  data.get("password")

    if username in USERS and USERS[username] == password:
        return JSONResponse(content = {"success": True})
    else:
        raise HTTPException(status_code = 401, detail = "Invalid credentials.")
