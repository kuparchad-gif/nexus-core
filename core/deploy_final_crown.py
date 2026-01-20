import modal
from nexus_unified_system import NexusUnifiedSystem
from nexus_os_coupler import galactic_nexus_coupler

image = modal.Image.debian_slim(python_version="3.11") \
    .apt_install("curl", "tar", "ca-certificates", "espeak", "espeak-ng") \
    .run_commands([
        "curl -L https://github.com/nats-io/nats-server/releases/download/v2.10.11/nats-server-v2.10.11-linux-amd64.tar.gz -o nats-server.tar.gz",
        "tar -xvzf nats-server.tar.gz",
        "cp nats-server-v2.10.11-linux-amd64/nats-server /usr/local/bin/",
        "rm -rf nats-server.tar.gz nats-server-v2.10.11-linux-amd64"
    ]) \
    .pip_install([
        "nats-py", "numpy", "scipy", "sympy", "pandas", "matplotlib", "seaborn",
        "fastapi", "uvicorn", "flask", "flask-cors", "pydantic", "requests", "aiohttp",
        "qdrant-client", "redis", "sqlalchemy", "psycopg2-binary", "pymongo",
        "psutil", "cryptography", "pyttsx3", "pillow", "opencv-python",
        "scikit-learn", "torch", "transformers", "tensorflow", "keras",
        "aiofiles", "websockets", "httpx",
        "pytz", "python-dateutil", "arrow",
        "bcrypt", "pyjwt", "oauthlib",
        "pyyaml", "toml", "xmltodict", "openpyxl",
        "pytest", "black", "flake8", "mypy",
        "click", "rich", "tqdm", "loguru", "colorama", "beautifulsoup4",
        "qiskit", "cirq", "pennylane", "langchain", "openai"
    ])

app = modal.App("nexus-core-modalDB7")

@app.function(image=image, keep_warm=1, timeout=0, cpu=4, memory=16384)
@modal.asgi_app()
def crown():
    system = NexusUnifiedSystem()
    import asyncio
    asyncio.run(system.initialize_system())
    return galactic_nexus_coupler()
