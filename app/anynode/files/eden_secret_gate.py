
import os
from google.cloud import secretmanager

def get_secret_from_gsm(secret_id = "eden-key"):
    try:
        client  =  secretmanager.SecretManagerServiceClient()
        name  =  f"projects/YOUR_PROJECT_ID/secrets/{secret_id}/versions/latest"
        response  =  client.access_secret_version(request = {"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return None

def validate_secret_route(auth_token):
    # 1. Try Google Secret Manager
    gsm_secret  =  get_secret_from_gsm()
    if gsm_secret and auth_token == gsm_secret:
        return True

    # 2. Try environment variable
    env_secret  =  os.getenv("EDEN_KEY")
    if env_secret and auth_token == env_secret:
        return True

    # 3. Hardcoded (as final fallback - avoid using in production)
    hardcoded_fallback  =  "OpenSesameEden"
    if auth_token == hardcoded_fallback:
        return True

    return False
