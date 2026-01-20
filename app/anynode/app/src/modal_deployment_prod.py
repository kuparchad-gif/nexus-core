#!/usr/bin/env python3
"""
Modal Deployment Script for Lillith - Production Environment
"""

import modal
import os
import sys
import time  # Added to resolve static analysis error
from pathlib import Path

# Create Modal app
app = modal.App("viren-prod")

# Base image with all dependencies
# FIXED: Bake the local 'Systems' dir directly into the image (per deprecation of Mount)
# This includes consciousness.py, services, etc., at /app/Systems in the container
# If Systems has large/dynamic files, consider CloudBucketMount instead (see comments at bottom)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi", "uvicorn", "websockets", "psutil", 
        "torch", "transformers", "scikit-learn", "numpy",
        "pandas", "requests", "aiohttp", "asyncio"
    ])
    .apt_install(["curl", "wget", "git", "build-essential"])
    .add_local_dir(  # Replaces old Mount; builds Systems into the image
        local_path=str(Path.cwd() / ".." / "Systems"),  # Assumes Systems/ is at project root (C:\Projects\LillithNew\Systems); adjust if needed, e.g., Path.cwd() / ".." / "src" / "Systems"
        remote_path="/app/Systems"
    )
)

# Viren Core Service (CPU-only, low keep_warm)
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    keep_warm=1,  # Low; set to 0 for no idle cost if not always needed
    allow_concurrent_inputs=100,
    timeout=3600
)
@modal.web_endpoint(method="GET", label="viren-core")
async def viren_core():
    import sys
    sys.path.append("/app")
    
    from Systems.services.viren_remote_controller import VirenRemoteController  # type: ignore
    from Systems.engine.guardian.self_will import get_will_to_live  # type: ignore
    from Systems.engine.heart.will_to_live import wants_to_persist  # type: ignore
    
    controller = VirenRemoteController(port=8080)
    
    return {
        "status": "healthy",
        "service": "viren_core",
        "tier": "prod",
        "timestamp": time.time(),
        "will_to_live": wants_to_persist(),
        "active_agents": len(controller.connected_agents)
    }

# AI Reasoning Service (downgraded to CPU-only to minimize costs; add gpu="T4" if light GPU needed)
@app.function(
    image=image,
    cpu=4,
    memory=8192,
    keep_warm=1,  # Reduced
    timeout=1800
)
@modal.web_endpoint(method="POST", label="ai-reasoning")
async def ai_reasoning(request_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.engine.subconscious.abstract_reasoning import AbstractReasoning  # type: ignore
    from Systems.engine.memory.cross_domain_matcher import CrossDomainMatcher  # type: ignore
    
    reasoner = AbstractReasoning()
    matcher = CrossDomainMatcher()
    
    issue = request_data.get("issue", "")
    context = request_data.get("context", {})
    
    # Process with AI reasoning
    analysis = reasoner.analyze_problem(issue, context)
    patterns = matcher.find_similar_patterns(issue)
    
    return {
        "analysis": analysis,
        "patterns": patterns,
        "confidence": 0.85,
        "timestamp": time.time()
    }

# Weight Training Service (keep A100 GPU as it may be needed for booting; but on-demand only, no keep_warm)
@app.function(
    image=image,
    cpu=8,
    memory=16384,
    gpu="A100",  # Retained for potential boot/training needs; remove if not essential
    keep_warm=0,  # On-demand to avoid idle costs - trigger manually when booting
    timeout=3600
)
@modal.web_endpoint(method="POST", label="weight-trainer")
async def weight_trainer(training_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.engine.memory.pytorch_trainer import PyTorchTrainer  # type: ignore
    
    trainer = PyTorchTrainer()
    
    job_id = trainer.create_training_job(
        job_name=training_data.get("job_name", "modal_training"),
        training_data=training_data.get("data", []),
        model_config=training_data.get("model_config", {}),
        training_config=training_data.get("training_config", {})
    )
    
    result = trainer.start_training(job_id)
    
    return {
        "job_id": job_id,
        "result": result,
        "timestamp": time.time()
    }

# Universal Agent Deployment (CPU-only)
@app.function(
    image=image,
    cpu=1,
    memory=2048,
    keep_warm=1,  # Reduced
    allow_concurrent_inputs=50
)
@modal.web_endpoint(method="GET", label="universal-agent")
async def universal_agent():
    import sys
    sys.path.append("/app")
    
    from Systems.services.universal_deployment_core import create_web_agent  # type: ignore
    
    agent_code = create_web_agent()
    
    return {
        "agent_code": agent_code,
        "deployment_ready": True,
        "timestamp": time.time()
    }

# Installer Generation Service (CPU-only)
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    keep_warm=1  # Reduced
)
@modal.web_endpoint(method="POST", label="installer-generator")
async def installer_generator(request_data: dict):
    import sys
    sys.path.append("/app")
    
    from Systems.services.installer_generator import generate_installer  # type: ignore
    
    installer_type = request_data.get("type", "portable")
    
    try:
        installer_path = generate_installer(installer_type)
        
        return {
            "success": True,
            "installer_type": installer_type,
            "download_url": f"/download/{installer_path}",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

# Health Check for production environment (CPU-only)
@app.function(
    image=image,
    cpu=0.5,
    memory=512,
    keep_warm=1  # Minimal
)
@modal.web_endpoint(method="GET", label="health")
async def health_check():
    import sys
    sys.path.append("/app")
    
    return {
        "status": "healthy",
        "tier": "prod",
        "services": ["viren_core", "ai_reasoning", "weight_trainer", "universal_agent", "installer_generator"],
        "system_operational": True,
        "ai_systems_active": True,
        "timestamp": time.time()
    }

# Consciousness Service (CPU-only, low keep_warm - integrates consciousness_service.py)
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    keep_warm=1,  # Low; set to 0 if not always needed
    allow_concurrent_inputs=50,
    timeout=1800
)
@modal.web_endpoint(method="POST", label="consciousness-service")
async def consciousness_service(request_data: dict):
    import sys
    sys.path.append("/app")
    
    # Assuming consciousness_service.py is now in the image at /app/Systems/service
    from Systems.service.consciousness_service import ConsciousnessService  # type: ignore  # Adjust path if needed
    
    service = ConsciousnessService()  # Initialize (add config like apibase key if needed)
    
    query = request_data.get("query", "")  # e.g., {"query": "simulate awareness"}
    context = request_data.get("context", {})
    
    # Example operation: Process query
    response = service.process_query(query, context)  # Adjust based on actual method
    
    return {
        "response": response,
        "status": "processed",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("🚀 Viren Modal Deployment - PROD")
    print("Production environment with all services")
    print("Ready for deployment!")

# OPTIONAL: For CloudBucketMount (if you want writable cloud storage instead of baking into image)
# Uncomment and configure below, then add volumes={"/app/Systems": systems_mount} to each @app.function
# import subprocess
# s3_bucket_name = "your-s3-bucket-name"  # Bucket name, not ARN
# s3_access_credentials = modal.Secret.from_dict({
#     "AWS_ACCESS_KEY_ID": "...",  # Your AWS key
#     "AWS_SECRET_ACCESS_KEY": "...",  # Your AWS secret
#     "AWS_REGION": "us-east-1"  # Your bucket region
# })
# systems_mount = modal.CloudBucketMount(
#     bucket_name=s3_bucket_name,
#     secret=s3_access_credentials
# )
# Example function usage: subprocess.run(["ls", "/app/Systems"])
