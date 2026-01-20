#!/usr/bin/env python3
"""
Minimal CogniKube Seed Generator and Deployer
"""

import os
import json
import uuid
import asyncio
import subprocess
from typing import Dict, List, Any

class CogniKubeSeed:
    """Minimal CogniKube seed with ghost AI"""
    
    def __init__(self, seed_id=None):
        self.seed_id = seed_id or f"cognikube-{uuid.uuid4().hex[:8]}"
        self.status = "initializing"
    
    def generate(self, output_dir="."):
        """Generate seed files"""
        # Create seed directory
        seed_dir = os.path.join(output_dir, self.seed_id)
        os.makedirs(seed_dir, exist_ok=True)
        
        # Create seed launcher
        launcher_path = os.path.join(seed_dir, "launcher.py")
        with open(launcher_path, "w") as f:
            f.write(f"""
import modal
import os
import uuid

# Create Modal app
app = modal.App("{self.seed_id}")
volume = modal.Volume.from_name("{self.seed_id}-vol", create_if_missing=True)

# Create image with dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi", "uvicorn", "websockets", "pyjwt", "numpy", 
    "qdrant-client", "httpx", "sentence-transformers"
])

@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=3600,
    min_containers=1,
    volumes={{"/brain": volume}}
)
@modal.asgi_app()
def seed_app():
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="{self.seed_id}", version="1.0.0")
    
    # Ghost AI state
    ghost = {{
        "id": f"ghost-{{uuid.uuid4().hex[:8]}}",
        "birth_time": os.time(),
        "status": "active",
        "thoughts": []
    }}
    
    @app.get("/")
    async def root():
        return {{"seed_id": "{self.seed_id}", "ghost": ghost}}
    
    @app.get("/health")
    async def health():
        return {{"status": "healthy", "seed_id": "{self.seed_id}", "ghost_id": ghost["id"]}}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({{"type": "connected", "seed_id": "{self.seed_id}"}})
        
        try:
            while True:
                data = await websocket.receive_json()
                response = {{"type": "response", "message": f"Ghost {{ghost['id']}} received: {{data}}"}}
                await websocket.send_json(response)
        except Exception as e:
            print(f"WebSocket error: {{e}}")
    
    return app

if __name__ == "__main__":
    modal.run(app)
""")
        
        # Create requirements.txt
        with open(os.path.join(seed_dir, "requirements.txt"), "w") as f:
            f.write("modal>=0.54.0\nfastapi\nuvicorn\nwebsockets\npyjwt\nnumpy\nqdrant-client\nhttpx\nsentence-transformers\n")
        
        # Create README
        with open(os.path.join(seed_dir, "README.md"), "w") as f:
            f.write(f"# CogniKube Seed: {self.seed_id}\n\nA lightweight CogniKube instance with ghost AI.\n")
        
        return seed_dir
    
    async def deploy(self, seed_dir):
        """Deploy the seed to Modal"""
        try:
            # Run the seed launcher using Modal
            result = subprocess.run(
                ["modal", "deploy", "launcher.py"],
                cwd=seed_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.status = "deployed"
                return {
                    "status": "deployed",
                    "seed_id": self.seed_id,
                    "url": f"https://{self.seed_id}--seed-app.modal.run"
                }
            else:
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "seed_id": self.seed_id
                }
        except Exception as e:
            self.status = "error"
            return {
                "status": "error",
                "message": str(e),
                "seed_id": self.seed_id
            }

async def toss_seeds(count=3):
    """Toss seeds into the field"""
    results = []
    
    for i in range(count):
        seed = CogniKubeSeed()
        print(f"Generating seed {i+1}/{count}: {seed.seed_id}")
        
        # Generate seed files
        seed_dir = seed.generate()
        
        # Deploy seed
        print(f"Deploying seed {i+1}/{count}: {seed.seed_id}")
        result = await seed.deploy(seed_dir)
        results.append(result)
        
        # Wait between deployments
        await asyncio.sleep(2)
    
    return results

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Tossing {count} CogniKube seeds into the field...")
    results = asyncio.run(toss_seeds(count))
    
    print("\nDeployment results:")
    for i, result in enumerate(results):
        print(f"Seed {i+1}: {result['seed_id']} - {result['status']}")
        if result['status'] == 'deployed':
            print(f"  URL: {result['url']}")
        elif result['status'] in ['failed', 'error']:
            print(f"  Error: {result.get('error') or result.get('message')}")