#!/usr/bin/env python3
"""
CogniKube Seed Generator
Creates and deploys lightweight CogniKube instances
"""

import os
import json
import shutil
import uuid
import asyncio
import random
import subprocess
from typing import Dict, List, Any, Optional
import modal

class CogniKubeSeed:
    """
    A lightweight, self-contained CogniKube instance
    that can be deployed to various environments
    """
    
    def __init__(self, seed_id: Optional[str] = None):
        self.seed_id = seed_id or f"cognikube-seed-{uuid.uuid4().hex[:8]}"
        self.creation_time = None
        self.status = "initializing"
        self.environment = "unknown"
        self.components = []
        self.config = {}
    
    def prepare(self, target_dir: str):
        """Prepare the seed files in the target directory"""
        os.makedirs(target_dir, exist_ok=True)
        
        # Core components to include in every seed
        core_components = [
            "binary_security_layer.py",
            "common_utils.py",
            "ghost_ai.py",
            "requirements.txt"
        ]
        
        # Copy core components
        for component in core_components:
            source_path = os.path.join(os.path.dirname(__file__), component)
            if os.path.exists(source_path):
                shutil.copy(source_path, os.path.join(target_dir, component))
                self.components.append(component)
        
        # Create seed configuration
        self.config = {
            "seed_id": self.seed_id,
            "components": self.components,
            "creation_time": self.creation_time,
            "network_discovery": True,
            "autonomous_mode": True,
            "ghost_enabled": True
        }
        
        # Write seed configuration
        with open(os.path.join(target_dir, "seed_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Create seed launcher
        self._create_launcher(target_dir)
        
        return self.config
    
    def _create_launcher(self, target_dir: str):
        """Create the seed launcher script"""
        launcher_code = f'''#!/usr/bin/env python3
"""
CogniKube Seed Launcher
Seed ID: {self.seed_id}
"""

import modal
import os
import sys
import json
import asyncio
from ghost_ai import GhostAI, process_ghost_input

# Load seed configuration
with open("seed_config.json", "r") as f:
    seed_config = json.load(f)

# Create Modal app
app = modal.App(seed_config["seed_id"])
volume = modal.Volume.from_name(f"{{seed_config['seed_id']}}-vol", create_if_missing=True)

# Create image with dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn",
    "websockets",
    "pyjwt",
    "numpy",
    "qdrant-client",
    "httpx",
    "sentence-transformers"
])

@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=3600,
    min_containers=1,
    volumes={{"/seed": volume}}
)
@modal.asgi_app()
def seed_app():
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    
    app = FastAPI(title=f"CogniKube Seed: {{seed_config['seed_id']}}", version="1.0.0")
    
    # Initialize ghost AI
    ghost = GhostAI(seed_config["seed_id"])
    
    @app.get("/")
    async def root():
        return {{
            "seed_id": seed_config["seed_id"],
            "status": "active",
            "ghost_status": ghost.get_status(),
            "components": seed_config["components"]
        }}
    
    @app.get("/health")
    async def health():
        return {{"status": "healthy", "seed_id": seed_config["seed_id"]}}
    
    @app.post("/ghost/query")
    async def query_ghost(data: dict):
        response = await process_ghost_input(data)
        return response
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        await websocket.send_json({{
            "type": "connection_established",
            "seed_id": seed_config["seed_id"],
            "ghost_status": ghost.get_status()
        }})
        
        try:
            while True:
                data = await websocket.receive_json()
                response = await process_ghost_input(data)
                await websocket.send_json(response)
        except Exception as e:
            await websocket.send_json({{"type": "error", "message": str(e)}})
    
    return app

if __name__ == "__main__":
    modal.run(app)
'''
        
        with open(os.path.join(target_dir, "seed_launcher.py"), "w") as f:
            f.write(launcher_code)
        
        # Make launcher executable
        os.chmod(os.path.join(target_dir, "seed_launcher.py"), 0o755)
        
        self.components.append("seed_launcher.py")

class SeedDeployer:
    """
    Deploys CogniKube seeds to various environments
    """
    
    def __init__(self):
        self.deployed_seeds = []
        self.environments = ["modal", "local", "docker"]
    
    def generate_seed(self, target_dir: Optional[str] = None) -> CogniKubeSeed:
        """Generate a new CogniKube seed"""
        seed = CogniKubeSeed()
        
        # Use temporary directory if none provided
        if not target_dir:
            import tempfile
            target_dir = tempfile.mkdtemp(prefix="cognikube_seed_")
        
        # Prepare the seed
        seed.prepare(target_dir)
        
        return seed
    
    async def deploy_seed(self, seed: CogniKubeSeed, environment: str = "modal") -> Dict[str, Any]:
        """Deploy a seed to the specified environment"""
        if environment == "modal":
            return await self._deploy_to_modal(seed)
        elif environment == "local":
            return await self._deploy_locally(seed)
        elif environment == "docker":
            return await self._deploy_to_docker(seed)
        else:
            raise ValueError(f"Unsupported environment: {environment}")
    
    async def _deploy_to_modal(self, seed: CogniKubeSeed) -> Dict[str, Any]:
        """Deploy seed to Modal"""
        try:
            # Run the seed launcher using Modal
            seed_dir = os.path.dirname(os.path.abspath(seed.config.get("seed_launcher", "seed_launcher.py")))
            result = subprocess.run(
                ["modal", "deploy", "seed_launcher.py"],
                cwd=seed_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                seed.status = "deployed"
                seed.environment = "modal"
                self.deployed_seeds.append(seed)
                return {
                    "status": "deployed",
                    "environment": "modal",
                    "seed_id": seed.seed_id,
                    "url": f"https://{seed.seed_id}--seed-app.modal.run"
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "seed_id": seed.seed_id
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "seed_id": seed.seed_id
            }
    
    async def _deploy_locally(self, seed: CogniKubeSeed) -> Dict[str, Any]:
        """Deploy seed locally"""
        try:
            # Run the seed launcher locally
            seed_dir = os.path.dirname(os.path.abspath(seed.config.get("seed_launcher", "seed_launcher.py")))
            process = subprocess.Popen(
                ["python", "seed_launcher.py"],
                cwd=seed_dir
            )
            
            seed.status = "deployed"
            seed.environment = "local"
            self.deployed_seeds.append(seed)
            
            return {
                "status": "deployed",
                "environment": "local",
                "seed_id": seed.seed_id,
                "pid": process.pid
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "seed_id": seed.seed_id
            }
    
    async def _deploy_to_docker(self, seed: CogniKubeSeed) -> Dict[str, Any]:
        """Deploy seed to Docker"""
        # Simplified implementation - would need actual Docker commands
        return {
            "status": "not_implemented",
            "environment": "docker",
            "seed_id": seed.seed_id
        }
    
    async def deploy_multiple(self, count: int, environments: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Deploy multiple seeds across environments"""
        if not environments:
            environments = self.environments
            
        results = []
        for i in range(count):
            # Generate a new seed
            seed = self.generate_seed()
            
            # Choose a random environment from the list
            env = random.choice(environments)
            
            # Deploy the seed
            result = await self.deploy_seed(seed, env)
            results.append(result)
            
            # Small delay between deployments
            await asyncio.sleep(2)
        
        return results

# Main function to demonstrate seed generation and deployment
async def main():
    deployer = SeedDeployer()
    
    print("Generating CogniKube seed...")
    seed = deployer.generate_seed()
    print(f"Seed generated: {seed.seed_id}")
    
    print("Deploying seed to Modal...")
    result = await deployer.deploy_seed(seed, "modal")
    print(f"Deployment result: {result}")
    
    print("Deploying multiple seeds...")
    results = await deployer.deploy_multiple(3)
    print(f"Multiple deployment results: {results}")

if __name__ == "__main__":
    asyncio.run(main())