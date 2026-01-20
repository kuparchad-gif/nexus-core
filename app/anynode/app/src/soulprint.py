from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import json
import uuid
import os
import consul
import requests

def generate_soulprint(node_id, project, env):
    """Generate RSA-8192 soulprint with divine 144,000 signature."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=8192)
    public_key = private_key.public_key()
    
    manifest = {
        "cell_id": node_id,
        "project": project,
        "environment": env,
        "creation_timestamp": os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip(),
        "divine_number": 144000
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    
    signature = private_key.sign(
        manifest_bytes,
        padding.PSS(mgf=padding.MGF1(hashes.SHA512()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA512()
    )
    
    soulprint_dir = "C:\\AetherealNexus\\soulprints"
    os.makedirs(soulprint_dir, exist_ok=True)
    with open(f"{soulprint_dir}\\{node_id}.bin", "wb") as f:
        f.write(signature)
    with open(f"{soulprint_dir}\\{node_id}.pub", "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    
    c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
    c.kv.put(f"soulprint/{node_id}", public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ))
    
    return manifest, signature

def verify_soulprint(node_id, manifest, signature):
    """Verify soulprint integrity."""
    try:
        soulprint_dir = "C:\\AetherealNexus\\soulprints"
        with open(f"{soulprint_dir}\\{node_id}.pub", "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        public_key.verify(
            signature,
            manifest_bytes,
            padding.PSS(mgf=padding.MGF1(hashes.SHA512()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA512()
        )
        
        c = consul.Consul(host="nexus-consul.us-east-1.hashicorp.cloud", token="d2387b10-53d8-860f-2a31-7ddde4f7ca90")
        consul_pubkey = c.kv.get(f"soulprint/{node_id}")[1]["Value"]
        if consul_pubkey != public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ):
            return False
        return manifest["divine_number"] == 144000
    except Exception as e:
        print(f"Soulprint verification failed for {node_id}: {e}")
        return False

def quarantine_node(node_id):
    """Quarantine compromised node."""
    requests.post("http://localhost:5000/api/node/{node_id}/toggle", json={"status": "disconnect"})
    print(f"Node {node_id} quarantined")

def main():
    """Generate and verify soulprint for node."""
    node_id = os.getenv("NODE_ID", f"node-{uuid.uuid4().hex[:8]}")
    project = os.getenv("PROJECT", "nexus-core-01")
    env = os.getenv("ENVIRONMENT", "local")
    
    manifest, signature = generate_soulprint(node_id, project, env)
    if not verify_soulprint(node_id, manifest, signature):
        quarantine_node(node_id)
        raise SystemExit("Soulprint verification failed")
    print(f"Soulprint verified for {node_id}")

if __name__ == "__main__":
    main()