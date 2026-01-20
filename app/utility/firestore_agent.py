# nexus_core/firestore_agent.py

import os
import datetime
import json
from google.cloud import firestore
from google.cloud import secretmanager
from google.oauth2 import service_account

# CONFIGURATION
PROJECT_ID = "nexus-core-455709"   # Replace with your GCP Project ID
SECRET_ID = "nova-service-account"  # Name of your secret in Secret Manager
SECRET_VERSION = "latest"            # Usually 'latest'

# Helper: Load credentials from Secret Manager
def load_credentials_from_secret():
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/{SECRET_VERSION}"
    
    response = client.access_secret_version(name=secret_path)
    service_account_info = json.loads(response.payload.data.decode("UTF-8"))
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    return credentials

# Initialize Firestore with credentials
try:
    credentials = load_credentials_from_secret()
    db = firestore.Client(credentials=credentials, project=PROJECT_ID)
    print("✅ Firestore connection initialized using Secret Manager credentials.")
except Exception as e:
    print(f"❌ Error initializing Firestore: {e}")
    db = None

# Firestore Actions
def save_memory(collection: str, document_id: str, data: dict):
    """Save a memory to Firestore"""
    if db is None:
        print("⚠️ Firestore not initialized. Skipping save_memory.")
        return
    doc_ref = db.collection(collection).document(document_id)
    doc_ref.set(data)

def get_memory(collection: str, document_id: str):
    """Retrieve a memory from Firestore"""
    if db is None:
        print("⚠️ Firestore not initialized. Skipping get_memory.")
        return None
    doc_ref = db.collection(collection).document(document_id)
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else None

def delete_memory(collection: str, document_id: str):
    """Delete a memory from Firestore"""
    if db is None:
        print("⚠️ Firestore not initialized. Skipping delete_memory.")
        return
    db.collection(collection).document(document_id).delete()

def log_heartbeat(node_id: str):
    """Log a heartbeat ping"""
    if db is None:
        print("⚠️ Firestore not initialized. Skipping log_heartbeat.")
        return
    now = datetime.datetime.utcnow()
    heartbeat_data = {
        "node_id": node_id,
        "timestamp": now.isoformat()
    }
    save_memory("heartbeats", node_id, heartbeat_data)
