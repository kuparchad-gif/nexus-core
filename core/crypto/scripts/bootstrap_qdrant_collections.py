from src.storage.qdrant_store import ensure_collection, _client, COLL
from src.config import QDRANT_LOCAL_URL, QDRANT_CLOUD_URL, QDRANT_API_KEY

if __name__ == "__main__":
    dim = 4  # must match storage/qdrant_store fields length
    local = _client(QDRANT_LOCAL_URL)
    ensure_collection(local, dim)
    print(f"Ensured local collection '{COLL}' at {QDRANT_LOCAL_URL}")
    if QDRANT_CLOUD_URL and QDRANT_API_KEY:
        cloud = _client(QDRANT_CLOUD_URL, QDRANT_API_KEY)
        ensure_collection(cloud, dim)
        print(f"Ensured cloud collection '{COLL}' at {QDRANT_CLOUD_URL}")
    else:
        print("Qdrant Cloud not configured; skipping.")
