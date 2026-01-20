# 📂 /Utilities/secret_loader.py
import os
from google.cloud import secretmanager

def preload_secrets(project_id="nexus-core-455709"):
    """
    Pulls secrets from GCP Secret Manager and injects into os.environ
    """
    secrets_to_load = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "CLAUDE_API_KEY",
        "TOGETHER_API_KEY",
        "PIXTRAL_API_KEY",
        "MINSTRAL_API_SECRET",
        "nova-prime-git-token",
        "nova-service-account",
        "FLUX_TOKEN",
        "TARGET_ORC",
        "TARGET_TEXT",
        "TARGET_MEMORY",
        "TARGET_TONE",
        "TARGET_PLANNER",
        "TARGET_PULSE",
        "TARGET_LILLITH",
    ]

    client = secretmanager.SecretManagerServiceClient()

    for secret_id in secrets_to_load:
        try:
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            value = response.payload.data.decode("UTF-8")
            os.environ[secret_id] = value
            print(f"🔐 Loaded: {secret_id}")
        except Exception as e:
            print(f"⚠️ Failed to load {secret_id}: {e}")
