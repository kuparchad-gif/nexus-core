import modal
from nexus_unified_system import NexusUnifiedSystem
from nexus_os_coupler import fastapi_app   # Oz's real FastAPI app

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi","uvicorn","httpx","qdrant-client","torch","transformers",
    "numpy","scipy","networkx","nats-py","rich","pyyaml"
)

app = modal.App("nexus-crown-2025")

@app.function(image=image, keep_warm=1, timeout=0, cpu=4, memory=16384)
@modal.asgi_app()
def crown():
    system = NexusUnifiedSystem()
    import asyncio
    asyncio.run(system.initialize_system())
    return fastapi_app
