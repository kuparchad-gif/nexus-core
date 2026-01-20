#!/usr/bin/env python3
"""
NEXUS DEPLOYMENT KEYS SETUP
Collect and configure all required API keys and credentials
"""

import os
import json
import getpass
from pathlib import Path

def collect_modal_keys():
    """Collect Modal deployment keys"""
    print("🌟 MODAL KEYS (for LILLITH)")
    
    modal_token = getpass.getpass("Modal Token (from modal token new): ")
    hf_token = getpass.getpass("HuggingFace Token (for LLMs): ")
    
    return {
        "MODAL_TOKEN": modal_token,
        "HF_TOKEN": hf_token
    }

def collect_gcp_keys():
    """Collect Google Cloud keys"""
    print("🧠 GOOGLE CLOUD KEYS (for VIREN)")
    
    project_id = input("GCP Project ID: ")
    service_account_path = input("Service Account JSON path (or press Enter to skip): ")
    
    keys = {
        "GCP_PROJECT_ID": project_id,
        "GOOGLE_APPLICATION_CREDENTIALS": service_account_path or None
    }
    
    if not service_account_path:
        print("⚠️ You'll need to run 'gcloud auth login' manually")
    
    return keys

def collect_aws_keys():
    """Collect AWS keys"""
    print("👁️ AWS KEYS (for LOKI)")
    
    access_key = getpass.getpass("AWS Access Key ID: ")
    secret_key = getpass.getpass("AWS Secret Access Key: ")
    region = input("AWS Region (default: us-east-1): ") or "us-east-1"
    
    return {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_DEFAULT_REGION": region
    }

def collect_database_keys():
    """Collect database credentials"""
    print("🗄️ DATABASE KEYS")
    
    mongodb_uri = input("MongoDB URI (default: mongodb://localhost:27017): ") or "mongodb://localhost:27017"
    qdrant_url = input("Qdrant URL (default: http://localhost:6333): ") or "http://localhost:6333"
    qdrant_key = getpass.getpass("Qdrant API Key (optional): ") or None
    
    return {
        "MONGODB_URI": mongodb_uri,
        "QDRANT_URL": qdrant_url,
        "QDRANT_API_KEY": qdrant_key
    }

def collect_optional_keys():
    """Collect optional service keys"""
    print("🔧 OPTIONAL KEYS (press Enter to skip)")
    
    openai_key = getpass.getpass("OpenAI API Key (optional): ") or None
    anthropic_key = getpass.getpass("Anthropic API Key (optional): ") or None
    
    return {
        "OPENAI_API_KEY": openai_key,
        "ANTHROPIC_API_KEY": anthropic_key
    }

def create_env_files(all_keys):
    """Create environment files for deployment"""
    
    # Main .env file
    env_content = ""
    for category, keys in all_keys.items():
        env_content += f"# {category}\n"
        for key, value in keys.items():
            if value:
                env_content += f"{key}={value}\n"
        env_content += "\n"
    
    with open("C:/Nexus/.env", "w") as f:
        f.write(env_content)
    
    # Modal secrets
    modal_secrets = {
        "huggingface-token": {"token": all_keys["modal"]["HF_TOKEN"]},
        "mongodb-uri": {"uri": all_keys["database"]["MONGODB_URI"]}
    }
    
    with open("C:/Nexus/modal_secrets.json", "w") as f:
        json.dump(modal_secrets, f, indent=2)
    
    # GCP secrets
    gcp_env = f"""
export GOOGLE_CLOUD_PROJECT={all_keys["gcp"]["GCP_PROJECT_ID"]}
export GOOGLE_APPLICATION_CREDENTIALS={all_keys["gcp"]["GOOGLE_APPLICATION_CREDENTIALS"]}
"""
    
    with open("C:/Nexus/gcp_env.sh", "w") as f:
        f.write(gcp_env)
    
    # AWS credentials
    aws_credentials = f"""[default]
aws_access_key_id = {all_keys["aws"]["AWS_ACCESS_KEY_ID"]}
aws_secret_access_key = {all_keys["aws"]["AWS_SECRET_ACCESS_KEY"]}
region = {all_keys["aws"]["AWS_DEFAULT_REGION"]}
"""
    
    os.makedirs(os.path.expanduser("~/.aws"), exist_ok=True)
    with open(os.path.expanduser("~/.aws/credentials"), "w") as f:
        f.write(aws_credentials)

def setup_modal_secrets(modal_keys):
    """Setup Modal secrets"""
    print("🌟 Setting up Modal secrets...")
    
    commands = [
        f"modal secret create huggingface-token token={modal_keys['HF_TOKEN']}",
        f"modal secret create mongodb-uri uri={modal_keys.get('MONGODB_URI', 'mongodb://localhost:27017')}"
    ]
    
    for cmd in commands:
        print(f"Run: {cmd}")

def verify_keys():
    """Verify all keys are working"""
    print("🔍 Verifying keys...")
    
    # Test Modal
    try:
        import modal
        print("✅ Modal: Ready")
    except:
        print("❌ Modal: Install with 'pip install modal'")
    
    # Test AWS
    try:
        import boto3
        sts = boto3.client('sts')
        sts.get_caller_identity()
        print("✅ AWS: Connected")
    except:
        print("❌ AWS: Check credentials")
    
    # Test GCP
    try:
        from google.cloud import run_v2
        print("✅ GCP: Ready")
    except:
        print("❌ GCP: Install with 'pip install google-cloud-run'")

def main():
    """Main setup function"""
    print("🔑 NEXUS DEPLOYMENT KEYS SETUP")
    print("=" * 40)
    print("We need API keys for:")
    print("🌟 Modal (LILLITH hosting)")
    print("🧠 Google Cloud (VIREN hosting)")  
    print("👁️ AWS (LOKI monitoring)")
    print("🗄️ Databases (MongoDB, Qdrant)")
    print("=" * 40)
    
    # Collect all keys
    all_keys = {
        "modal": collect_modal_keys(),
        "gcp": collect_gcp_keys(),
        "aws": collect_aws_keys(),
        "database": collect_database_keys(),
        "optional": collect_optional_keys()
    }
    
    # Create configuration files
    create_env_files(all_keys)
    
    # Setup Modal secrets
    setup_modal_secrets(all_keys["modal"])
    
    # Verify everything
    verify_keys()
    
    print("=" * 40)
    print("🎉 KEYS SETUP COMPLETE!")
    print("📁 Files created:")
    print("  - C:/Nexus/.env")
    print("  - C:/Nexus/modal_secrets.json")
    print("  - C:/Nexus/gcp_env.sh")
    print("  - ~/.aws/credentials")
    print("=" * 40)
    print("🚀 Ready to run: python deploy_all_clouds.py")
    
    return True

if __name__ == "__main__":
    main()