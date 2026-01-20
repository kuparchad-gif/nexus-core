import modal
import os
from pathlib import Path

app = modal.App("nexus-core-mod-DB7")

def should_include(path: Path) -> bool:
    """DIAGNOSTIC FILTER - Find AND fix issues"""
    
    # ONLY exclude proven problematic directory
    if any(part in ['Anynode'] for part in path.parts):
        print(f"🚫 FILTERED: {path}") 
        return False
    
    # Log important files being included
    if path.is_file():
        if "nexus_unified_system.py" in str(path):
            print(f"✅ INCLUDING CRITICAL FILE: {path}")
        elif path.suffix in ['.py', '.bin']:
            print(f"📁 INCLUDING: {path}")
    
    # FILTER large files that might cause issues (but log them first)
    if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        print(f"🚫 FILTERING LARGE FILE: {path} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
        return False
    
    # FILTER specific file types (but log them)
    excluded_extensions = {'.gguf', '.zip', '.tar.gz', '.7z', '.rar', '.mp4', '.avi', '.mov'}
    if path.suffix in excluded_extensions:
        print(f"🚫 FILTERING EXTENSION: {path}")
        return False
    
    return True  # Include everything else

# Build image with filtered file copy
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "tar", "ca-certificates", "espeak", "espeak-ng")
    .run_commands(
        "curl -L https://github.com/nats-io/nats-server/releases/download/v2.10.11/nats-server-v2.10.11-linux-amd64.tar.gz -o /tmp/nats.tar.gz",
        "tar -xzf /tmp/nats.tar.gz -C /tmp",
        "cp /tmp/nats-server-v2.10.11-linux-amd64/nats-server /usr/local/bin/",
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
        "tiktoken", "prometheus_client", "structlog", "flwr"
    )
    .add_local_dir(
        local_path="C://project-root//10_env//aethereal_nexus",
        remote_path="/",
        ignore=should_include,
        copy=True
    )
    # Copy Core directory explicitly
    .add_local_dir(
        local_path="C://project-root//10_env//aethereal_nexus/core",
        remote_path="/core",
        ignore=should_include,
        copy=True
    )
    # Copy App directory explicitly  
    .add_local_dir(
        local_path="C://project-root//10_env//aethereal_nexus/app",
        remote_path="/app", 
        ignore=should_include,
        copy=True
    )
    # Copy Config directory explicitly
    .add_local_dir(
        local_path="C://project-root//10_env//aethereal_nexus/config",
        remote_path="/config",
        ignore=should_include,
        copy=True
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
    
    import sys
    import os
    
    # Debug what files actually made it
    root_files = [f for f in os.listdir("/") if not f.startswith(('.', '__'))]
    print(f"Root contents ({len(root_files)} items): {root_files}")
    
    # Check if core directory exists and what's in it
    if os.path.exists("/core"):
        core_files = os.listdir("/core")
        print(f"Core directory exists with {len(core_files)} files")
        if "nexus_unified_system.py" in core_files:
            print("✅ nexus_unified_system.py FOUND!")
        else:
            print("❌ nexus_unified_system.py MISSING!")
    else:
        print("❌ Core directory does not exist!")
        return {"error": "Core files not deployed"}
    
    # Set up Python path - SIMPLE VERSION
    sys.path.insert(0, "/")
    
    try:
        # Try the imports
        from core.nexus_unified_system import NexusUnifiedSystem
        from core.nexus_os_coupler import galactic_nexus_coupler
        
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("🚀 STARTING NEXUS SYSTEM...")
        
        system = NexusUnifiedSystem()
        import asyncio
        asyncio.run(system.initialize_system())
        return galactic_nexus_coupler()
        
    except Exception as e:
        print(f"❌ SYSTEM STARTUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Startup failed: {str(e)}"}
