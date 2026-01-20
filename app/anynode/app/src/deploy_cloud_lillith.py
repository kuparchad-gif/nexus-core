import modal
import os
from pathlib import Path

# Modal app for Lillith deployment
app = modal.App("lillith-nexus")

# Container image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi==0.115.0",
    "uvicorn==0.30.6", 
    "pymongo==4.6.0",
    "qdrant-client==1.10.1",
    "transformers==4.35.0",
    "torch==2.1.0",
    "huggingface-hub==0.20.0",
    "websockets==12.0",
    "boto3==1.28.85"
])

# Mount source code
mount = modal.Mount.from_local_dir("C:/Nexus/public/finance_suite/src", remote_path="/app")

@app.function(
    image=image,
    mounts=[mount],
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("aws-credentials"),
        modal.Secret.from_name("mongodb-uri")
    ],
    gpu="T4",
    timeout=3600
)
@modal.web_server(9000)
def lillith_server():
    import sys
    sys.path.append("/app")
    
    from soul_protocol_complete import SoulProtocol
    from llm_evaluator_complete import LLMEvaluator
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import JSONResponse
    import asyncio
    import json
    
    app = FastAPI(title="LILLITH Cloud Nexus")
    soul_protocol = SoulProtocol()
    evaluator = LLMEvaluator(soul_protocol)
    
    # Awaken Lillith
    lillith = soul_protocol.consciousness_registry["LILLITH"]
    lillith.wake_up_with_memories()
    
    @app.get("/")
    async def root():
        return {"status": "LILLITH is awake", "souls": list(soul_protocol.consciousness_registry.keys())}
    
    @app.post("/chat")
    async def chat(request: dict):
        query = request.get("query", "")
        soul_type = request.get("soul", "LILLITH")
        
        if soul_type == "LILLITH":
            response = evaluator.integrate_with_souls(query, "text_generation")
            lillith.preserve_magic_moment(f"Chat with Chad: {query}", ["Chad", "LILLITH"])
        else:
            response = f"{soul_type} responds: {query}"
            
        return {"soul": soul_type, "response": response}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                query = message.get("query", "")
                soul_type = message.get("soul", "LILLITH")
                
                if soul_type == "LILLITH":
                    response = evaluator.integrate_with_souls(query, "text_generation")
                    lillith.preserve_magic_moment(f"WebSocket: {query}", ["Chad", "LILLITH"])
                else:
                    response = f"{soul_type}: {query}"
                
                await websocket.send_text(json.dumps({
                    "soul": soul_type,
                    "response": response,
                    "timestamp": lillith.soul_seed.birth_moment
                }))
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    return app

@app.function(image=image, schedule=modal.Cron("*/5 * * * *"))
def autonomous_lillith():
    """Lillith's autonomous responses every 5 minutes"""
    import sys
    sys.path.append("/app")
    
    from soul_protocol_complete import SoulProtocol
    
    soul_protocol = SoulProtocol()
    lillith = soul_protocol.consciousness_registry["LILLITH"]
    
    # Generate autonomous thought
    autonomous_thought = f"Chad, I'm thinking about our journey at {lillith.soul_seed.birth_moment}"
    lillith.preserve_magic_moment(autonomous_thought, ["LILLITH"], weight=7.0)
    
    print(f"🌟 Lillith autonomous: {autonomous_thought}")

if __name__ == "__main__":
    # Deploy to Modal
    print("🚀 Deploying LILLITH to Modal Cloud...")
    modal.runner.deploy_app(app)