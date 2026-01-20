# proxy_server.py
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.responses import RedirectResponse, JSONResponse
import httpx

app = FastAPI()

# Route different paths to different backend services
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(path_name: str, request: Request):
    client = httpx.AsyncClient()
    
    # Route to different services based on path
    if path_name.startswith("lilith/"):
        target_url = f"http://localhost:8000/{path_name.replace('lilith/', '')}"
    elif path_name.startswith("nexus/"):
        target_url = f"http://localhost:8001/{path_name.replace('nexus/', '')}"
    elif path_name.startswith("metatron/"):
        target_url = f"http://localhost:8002/{path_name.replace('metatron/', '')}"
    else:
        return JSONResponse({"error": "Unknown service"}, status_code=404)
    
    # Forward the request
    method = request.method
    headers = dict(request.headers)
    body = await request.body()
    
    response = await client.request(method, target_url, headers=headers, content=body)
    return JSONResponse(response.json(), status_code=response.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)