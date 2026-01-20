import os
import subprocess
from huggingface_hub import snapshot_download
from qdrant_client import QdrantClient
from datetime import datetime

def deploy_sqlcoder_to_qdrant(qdrant_hosts, hf_token):
    """Clone and deploy SQLCoder-8B to each Qdrant instance."""
    model_repo = "defog/llama-3-sqlcoder-8b"
    model_dir = "./sqlcoder-8b"
    
    # Ensure git-lfs is installed
    subprocess.run(["git", "lfs", "install"], check=True)
    
    # Clone model using Hugging Face token
    os.environ["HF_TOKEN"] = hf_token
    snapshot_download(repo_id=model_repo, local_dir=model_dir, local_dir_use_symlinks=False)
    
    # Configure each Qdrant instance
    for host in qdrant_hosts:
        client = QdrantClient(host=host, port=6333)
        # Store model metadata in Qdrant
        client.upload_collection(
            collection_name="llm_models",
            vectors=[[0.1] * 768],  # Placeholder vector
            payload={
                "model_id": model_repo,
                "model_path": f"{host}:{model_dir}",
                "capabilities": ["sql_generation"],
                "deployed_at": datetime.now().isoformat()
            }
        )
        print(f"🌟 Deployed SQLCoder-8B to Qdrant at {host}")

if __name__ == "__main__":
    qdrant_hosts = ["qdrant1", "qdrant2", "localhost"]  # Replace with your Qdrant hosts
    hf_token = os.getenv("HF_TOKEN") or "your_huggingface_token"
    deploy_sqlcoder_to_qdrant(qdrant_hosts, hf_token)