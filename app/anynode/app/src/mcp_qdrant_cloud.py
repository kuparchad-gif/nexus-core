import modal
import os
from pathlib import Path
from typing import List, Dict, Any
import time
import functools

# Configuration
class Settings:
    QDRANT_COLLECTION = "nexus_files"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension
    MODEL_NAME = "all-MiniLM-L6-v2"
    MAX_FILE_BATCH = 100
    QDRANT_TIMEOUT = 30
    VOLUME_MOUNT_PATH = "/my_vol"
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1
    SERVER_TIMEOUT = 1800  # 30 minutes in seconds

# Modal setup
app = modal.App("mcp-server-production-final-fixed")
volume = modal.Volume.from_name("deployment-team-volume")

# Pinned dependency versions with fixed compatibility
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi==0.95.2",
        "uvicorn==0.22.0",
        "qdrant-client==1.6.3",
        # Fixed versions to resolve import conflicts
        "huggingface-hub==0.10.0",  # Compatible with sentence-transformers
        "sentence-transformers==2.2.2",
        "transformers==4.29.2",
        "torch==2.0.1",
        "numpy==1.24.3",
        "python-multipart==0.0.6"
    )
)

def retry_operation(max_attempts=Settings.RETRY_ATTEMPTS, delay=Settings.RETRY_DELAY):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
                    continue
            raise last_exception
        return wrapper
    return decorator

def get_fastapi_app():
    from fastapi import FastAPI, HTTPException, Body, status
    from fastapi.responses import FileResponse
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance, VectorParams
    from sentence_transformers import SentenceTransformer

    app = FastAPI(title="MCP Server", version="5.0.0")
    encoder = SentenceTransformer(Settings.MODEL_NAME, device='cpu')

    @app.get("/health", status_code=status.HTTP_200_OK)
    @retry_operation()
    async def health():
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            https=True,
            timeout=Settings.QDRANT_TIMEOUT
        )
        
        try:
            # Calculate volume size
            volume_status = 0
            for f in Path(Settings.VOLUME_MOUNT_PATH).rglob('*'):
                if f.is_file():
                    volume_status += f.stat().st_size
            
            # Check collection exists
            collections = client.get_collections()
            collection_exists = any(
                col.name == Settings.QDRANT_COLLECTION 
                for col in collections.collections
            )
            
            return {
                "status": "healthy",
                "components": {
                    "qdrant": "available",
                    "volume_size_bytes": volume_status,
                    "vector_count": collection_exists,
                    "cluster_id": "3df8b5df-91ae-4b41-b275-4c1130beed0f"
                }
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service degradation: {str(e)}"
            )

    @app.get("/list_files", status_code=status.HTTP_200_OK)
    async def list_files() -> Dict[str, List[str]]:
        volume.reload()
        files = []
        for root, _, filenames in os.walk(Settings.VOLUME_MOUNT_PATH):
            for filename in filenames:
                full_path = Path(root) / filename
                if full_path.is_file():
                    files.append(str(full_path.relative_to(Settings.VOLUME_MOUNT_PATH)))
        return {"files": files}

    @app.get("/read_file", status_code=status.HTTP_200_OK)
    async def read_file(filename: str) -> Dict[str, str]:
        safe_path = Path(Settings.VOLUME_MOUNT_PATH) / filename
        if not safe_path.resolve().is_relative_to(Path(Settings.VOLUME_MOUNT_PATH)):
            raise HTTPException(400, "Invalid file path")
        
        if not safe_path.is_file():
            raise HTTPException(404, "File not found")
            
        volume.reload()
        with open(safe_path, 'r', encoding='utf-8') as f:
            return {"content": f.read()}

    @app.get("/download", status_code=status.HTTP_200_OK)
    async def download_file(filename: str):
        """Download a file from the volume"""
        safe_path = Path(Settings.VOLUME_MOUNT_PATH) / filename
        if not safe_path.resolve().is_relative_to(Path(Settings.VOLUME_MOUNT_PATH)):
            raise HTTPException(400, "Invalid file path")
        if not safe_path.is_file():
            raise HTTPException(404, "File not found")
            
        volume.reload()
        return FileResponse(
            path=safe_path,
            filename=filename,
            media_type='application/octet-stream'
        )

    @app.post("/write_file", status_code=status.HTTP_201_CREATED)
    @retry_operation()
    async def write_file(
        filename: str = Body(..., embed=True),
        content: str = Body(..., embed=True)
    ) -> Dict[str, str]:
        safe_path = Path(Settings.VOLUME_MOUNT_PATH) / filename
        if not safe_path.resolve().is_relative_to(Path(Settings.VOLUME_MOUNT_PATH)):
            raise HTTPException(400, "Invalid file path")
            
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        volume.commit()
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            https=True,
            timeout=Settings.QDRANT_TIMEOUT
        )
        
        vector = encoder.encode(content).tolist()
        point = PointStruct(
            id=filename,
            vector=vector,
            payload={
                'name': filename,
                'path': str(safe_path),
                'size': len(content)
            }
        )
        
        client.upsert(
            collection_name=Settings.QDRANT_COLLECTION,
            points=[point]
        )
        
        return {"status": "file_written_and_vectorized"}

    @app.get("/search_files", status_code=status.HTTP_200_OK)
    @retry_operation()
    async def search_files(query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            https=True,
            timeout=Settings.QDRANT_TIMEOUT
        )
        
        vector = encoder.encode(query).tolist()
        results = client.search(
            collection_name=Settings.QDRANT_COLLECTION,
            query_vector=vector,
            limit=min(limit, 10)
        )
        
        return {"results": [hit.payload for hit in results]}

    return app

@app.function(
    image=image,
    volumes={Settings.VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("qdrant-secret")],
    timeout=Settings.SERVER_TIMEOUT  # 30 minutes timeout
)
@modal.asgi_app()
def asgi_app():
    return get_fastapi_app()

@app.function(
    image=image,
    volumes={Settings.VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("qdrant-secret")],
    timeout=60 * 15  # 15 minutes for startup
)
def startup():
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance, VectorParams
    from sentence_transformers import SentenceTransformer
    import zipfile

    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        https=True,
        timeout=Settings.QDRANT_TIMEOUT
    )
    
    encoder = SentenceTransformer(Settings.MODEL_NAME, device='cpu')
    
    # Check if collection exists by listing collections
    collections = client.get_collections()
    collection_exists = any(
        col.name == Settings.QDRANT_COLLECTION 
        for col in collections.collections
    )
    
    if not collection_exists:
        client.create_collection(
            collection_name=Settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=Settings.VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    
    # Process initial files if present
    if Path(f'{Settings.VOLUME_MOUNT_PATH}/files.zip').exists():
        with zipfile.ZipFile(f'{Settings.VOLUME_MOUNT_PATH}/files.zip', 'r') as zip_ref:
            zip_ref.extractall(Settings.VOLUME_MOUNT_PATH)
        volume.commit()
        
        points = []
        for root, _, files in os.walk(Settings.VOLUME_MOUNT_PATH):
            for file in files:
                if file == 'files.zip':
                    continue
                    
                path = Path(root) / file
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    vector = encoder.encode(content).tolist()
                    points.append(PointStruct(
                        id=file,
                        vector=vector,
                        payload={
                            'name': file,
                            'path': str(path),
                            'size': len(content)
                        }
                    ))
                    
                    if len(points) >= Settings.MAX_FILE_BATCH:
                        client.upsert(
                            collection_name=Settings.QDRANT_COLLECTION,
                            points=points
                        )
                        points = []
                        
                except Exception as e:
                    print(f"Skipping {file}: {str(e)}")
        
        if points:
            client.upsert(
                collection_name=Settings.QDRANT_COLLECTION,
                points=points
            )

@app.function(
    image=image,
    volumes={Settings.VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("qdrant-secret")],
    timeout=60 * 5  # 5 minutes for verification
)
def verify_deployment():
    """Comprehensive deployment verification"""
    from qdrant_client import QdrantClient
    
    # Test volume access
    test_file = Path(Settings.VOLUME_MOUNT_PATH) / "verification.txt"
    try:
        test_file.write_text("Verification successful")
        volume.commit()
        volume_content = test_file.read_text()
    except Exception as e:
        return {"status": "volume_failed", "error": str(e)}

    # Test Qdrant connection
    try:
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            https=True,
            timeout=5
        )
        collections = client.get_collections()
        collection_exists = any(
            col.name == Settings.QDRANT_COLLECTION 
            for col in collections.collections
        )
        qdrant_status = {
            "connected": True,
            "collections": [col.name for col in collections.collections],
            "target_collection_exists": collection_exists
        }
    except Exception as e:
        return {"status": "qdrant_failed", "error": str(e)}

    return {
        "status": "fully_operational",
        "volume": volume_content,
        "qdrant": qdrant_status,
        "cluster": "3df8b5df-91ae-4b41-b275-4c1130beed0f"
    }