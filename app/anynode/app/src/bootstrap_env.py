import os
import requests
import json
from secret_loader import preload_secrets

# 🌟 Metadata endpoints for GCP
METADATA_FLAVOR = {'Metadata-Flavor': 'Google'}
BASE_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/"

# 🌱 Environment variables to populate (leave placeholders)
ENV_VARS = [
    "ENVIRONMENT",
    "TOWER_ID",
    "TOWER_NAME",
    "PROJECT_ID",
    "REGION",
    "INSTANCE_ID",
    "PORT",
    "HEARTBEAT_INTERVAL_SEC",
    "PULSE_TRUSTED_PORT",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "CLAUDE_API_KEY",
    "TOGETHER_API_KEY",
    "PIXTRAL_API_KEY",
    "MINSTRAL_API_SECRET",
    "NOVA_GIT_TOKEN",
    "NOVA_SERVICE_ACCOUNT",
    "FLUX_TOKEN",
    "FLEET_LOGGER_ENABLED",
    "ALLOW_SELF_EVOLUTION"
]

# 🧹 Step 1: Clear .env

def clear_env():
    with open(".env", "w") as f:
        f.write("")

# 🔎 Step 2: Query metadata

def fetch_metadata(path):
    try:
        url = BASE_METADATA_URL + path
        response = requests.get(url, headers=METADATA_FLAVOR, timeout=3)
        if response.status_code == 200:
            return response.text
    except Exception:
        return None

# 🛠️ Step 3: Populate basic info

def populate_basic_info():
    environment = "production"
    project_id = fetch_metadata("project/project-id") or "unknown-project"
    region_zone = fetch_metadata("instance/zone")
    region = region_zone.split("/")[-1].rsplit('-', 1)[0] if region_zone else "unknown-region"
    instance_id = fetch_metadata("instance/id") or "unknown-instance"

    os.environ["ENVIRONMENT"] = environment
    os.environ["PROJECT_ID"] = project_id
    os.environ["REGION"] = region
    os.environ["INSTANCE_ID"] = instance_id
    os.environ["PORT"] = os.getenv("PORT", "8080")  # Safe default
    os.environ["HEARTBEAT_INTERVAL_SEC"] = "13"
    os.environ["PULSE_TRUSTED_PORT"] = "1313"
    os.environ["FLEET_LOGGER_ENABLED"] = "true"
    os.environ["ALLOW_SELF_EVOLUTION"] = "false"

# 🧪 Step 4: Fill gaps for missing names/IDs

def fill_defaults():
    if not os.getenv("TOWER_ID"):
        os.environ["TOWER_ID"] = "Trinity-" + os.urandom(4).hex()
    if not os.getenv("TOWER_NAME"):
        os.environ["TOWER_NAME"] = "Ernie-Temp-Tower"

# 📝 Step 5: Write final .env file

def write_env_file():
    with open(".env", "a") as f:
        for var in ENV_VARS:
            value = os.getenv(var, "")
            f.write(f"{var}={value}\n")

# 🚀 Bootstrap function

def bootstrap_environment():
    print("[ Trinity Bootstrapper ] Starting environment initialization...")
    clear_env()
    populate_basic_info()
    preload_secrets()
    fill_defaults()
    write_env_file()
    print("[ Trinity Bootstrapper ] Environment ready. Proceeding to boot...")

if __name__ == "__main__":
    bootstrap_environment()
