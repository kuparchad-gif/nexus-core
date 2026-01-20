import os
import yaml
from dotenv import load_dotenv

print("🧠 config.py loading...")

# Load environment variables from .env
load_dotenv()

# Primary secret token - unified name
FLUX_TOKEN = os.getenv("FLUX_TOKEN")

if not FLUX_TOKEN:
    print("⚠️ FLUX_TOKEN is not set in environment.")
else:
    print("✅ FLUX_TOKEN loaded.")

# Load Nova's runtime configuration
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../config/nova-config.yaml")

try:
    with open(CONFIG_FILE, "r") as file:
        NOVA_CONFIG = yaml.safe_load(file)
        print("✅ nova-config.yaml loaded.")
except Exception as e:
    NOVA_CONFIG = {}
    print(f"❌ Failed to load nova-config.yaml: {e}")

# Optional: Pull secrets from Secret Manager (disabled by default)
USE_SECRET_MANAGER = NOVA_CONFIG.get("use_secret_manager", True)

if USE_SECRET_MANAGER:
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()

        def access_secret(secret_name: str):
            project_id = os.getenv("GCP_PROJECT")
            if not project_id:
                print("❌ GCP_PROJECT not set in .env or environment.")
                return None
            full_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": full_path})
            return response.payload.data.decode("UTF-8")

        flux_from_secret = access_secret("flux_token")
        if flux_from_secret:
            FLUX_TOKEN = flux_from_secret
            print("✅ FLUX_TOKEN loaded from Secret Manager.")

    except Exception as e:
        print(f"❌ Error accessing Secret Manager: {e}")
        print("⚠️ Defaulting to .env FLUX_TOKEN.")

# Optional config flag access (example)
SELF_UPDATE_ENABLED = NOVA_CONFIG.get("self_update", False)
DREAMS_ENABLED = NOVA_CONFIG.get("dreams_enabled", False)
