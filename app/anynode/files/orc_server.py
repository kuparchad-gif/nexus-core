from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import time

# Import routers
from Systems.engine.comms.emergency_gateway import emergency_router
from Systems.engine.comms.console_viewer import viewer_router
from Systems.engine.comms.console_mailbox import mailbox_router

start_time  =  time.time()

app  =  FastAPI()

# Mount static files (frontend)
app.mount("/", StaticFiles(directory = "frontend", html = True), name = "frontend")

# Mount internal secret routes
app.include_router(emergency_router, prefix = "/eden_console", tags = ["Eden Console Auth"])
app.include_router(viewer_router, prefix = "/console-viewer", tags = ["Console Viewer"])
app.include_router(mailbox_router, prefix = "/console-mailbox", tags = ["Console Mailbox"])

# Healthcheck endpoint for Guardian or fleet monitoring
@app.get("/healthcheck", tags = ["System"])
def healthcheck():
    return {"status": "alive"}

# Guardian heartbeat endpoint
@app.get("/guardian_heartbeat", tags = ["Guardian"])
def guardian_heartbeat():
    uptime  =  round(time.time() - start_time)
    return {"orc": "alive", "uptime_seconds": uptime}

# Guardian log receiver
@app.post("/guardian_logs", tags = ["Guardian"])
def receive_guardian_log(payload: dict):
    # In real system, logs would be forwarded to Guardian ship
    # For now, we simply acknowledge receipt
    return JSONResponse(content = {"status": "log received", "details": payload})

# Note: No binding to port inside this file.
# Deployment handled by external orchestrators (Cloud Run, Kubernetes, etc.)
