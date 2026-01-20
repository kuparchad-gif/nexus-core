from fastapi import FastAPI
from social_intelligence_api import app as social_app
from cognikube_full import StandardizedPod, initialize_system

app = FastAPI()
app.mount("/api", social_app)

@app.on_event("startup")
async def startup_event():
    initialize_system()  # Initialize pods, Qdrant, etc.
    print("Lillith system started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)