from modal import App, Image, web_endpoint
import os

app = App("nexus1")
image = Image.debian_slim().pip_install([
    "fastapi", "uvicorn", "websockets", "openai", "transformers", 
    "torch", "numpy", "requests", "python-dotenv"
])

@app.function(image=image)
@web_endpoint()
def nexus1():
    import asyncio
    import websockets
    import json
    from datetime import datetime
    
    async def handle_websocket(websocket, path):
        async for message in websocket:
            if message == "tarot":
                response = f"Tarot Vision: A walk today brings a spark of inspiration - {datetime.now()}"
                await websocket.send(response)
    
    return {"status": "Nexus-1 Modal deployment active", "websocket": "ws://nexus1.modal.run/ws"}