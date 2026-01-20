import modal
import os
from pathlib import Path

app = modal.App("nexus-core-modalDB7")

def should_include(path: Path) -> bool:
    # Always include directories
    if path.is_dir():
        return True
    
    # Skip GGUF files and archives
    excluded_extensions = {'.gguf', '.zip', '.tar.gz', '.7z', '.rar'}
    if path.suffix in excluded_extensions:
        return False
    
    # Skip video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    if path.suffix in video_extensions:
        return False
    
    # Include everything else
    return True

# Build image with filtered file copy
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar", "ca-certificates", "espeak", "espeak-ng")
    .run_commands(
        "curl -L https://github.com/nats-io/nats-server/releases/download/v2.10.11/nats-server-v2.10.11-linux-amd64.tar.gz -o /tmp/nats.tar.gz",
        "tar -xzf /tmp/nats.tar.gz -C /tmp",
        "cp /tmp/nats-server-v2.10.11-linux-amd64/nats-server /usr/local/bin/",
        "cp -r /tmp_project_src/* / || true",  # Force recursive copy
        "rm -rf /tmp/nats*",
    )
    .uv_pip_install(
        "nats-py", "numpy", "scipy", "sympy", "pandas", "matplotlib", "seaborn",
        "fastapi", "uvicorn[standard]", "flask", "flask-cors", "pydantic", "requests", "aiohttp",
        "qdrant-client", "redis", "sqlalchemy", "psycopg2-binary", "pymongo",
        "psutil", "cryptography", "pyttsx3", "pillow", "opencv-python",
        "scikit-learn", "torch", "transformers", "tensorflow", "keras",
        "aiofiles", "websockets", "httpx", "networkx",
        "pytz", "python-dateutil", "arrow",
        "bcrypt", "pyjwt", "oauthlib", "ray", "langchain",
        "pyyaml", "toml", "xmltodict", "openpyxl",
        "pytest", "black", "flake8", "mypy",
        "click", "rich", "tqdm", "loguru", "colorama", "beautifulsoup4",
        "qiskit", "cirq", "pennylane", "langchain", "openai",
        "tiktoken", "prometheus_client", "structlog", "flwr",
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus",
    remote_path="/",
    copy=True,
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app",
    remote_path="/",
    copy=True,
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//crypto",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//OzOs",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//agents",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//qdrant_heroku_cli",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//skill_library",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//RevenueEngine",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//compressionEngine",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//MetatronValidation",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//models",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//SystemMonitor",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//TrainingOrchestrator",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training//universal models",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training_data",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//vc_pitch_engine",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//training",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//web",
    remote_path="/",
    copy=True,    
    )
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//core//system//webparts",
    remote_path="/",
    copy=True,    
    )
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app/acidemikubes",
    remote_path="/",
    copy=True,    
    )
     
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app//genesis_system",
    remote_path="/",
    copy=True,    
    )
     
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app//hermes-firmware",
    remote_path="/",
    copy=True,    
    )
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app//nexus-cognition",
    remote_path="/",
    copy=True,    
    )
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app//trinity_fx",
    remote_path="/",
    copy=True,    
    )
     
    .add_local_dir(
    local_path="C://project-root//10_env//aethereal_nexus//app//utility",
    remote_path="/",
    copy=True,    
    )
    
    
    
)

@app.function(
    image=image,
    min_containers=1,
    timeout=0,
    cpu=8,
    memory=16384,
    secrets=[modal.Secret.from_name("nexus-secrets")],
)
@modal.asgi_app()
def crown():
    print("=== LILLITH CROWN — USING ROOT ===")
    print("Root contents:", sorted(os.listdir("/")))
    
    import sys
    sys.path.insert(0, "/")
    
    from nexus_unified_system import NexusUnifiedSystem
    from nexus_os_coupler import galactic_nexus_coupler
    
    system = NexusUnifiedSystem()
    import asyncio
    asyncio.run(system.initialize_system())
    return galactic_nexus_coupler()