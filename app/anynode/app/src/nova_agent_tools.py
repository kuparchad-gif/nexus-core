# nova_engine/modules/executor/nova_agent_tools.py

import requests
import os

# 🔐 Load flux_token securely from config
from config import FLUX_TOKEN

# 🌐 Base URL for self or remote deployment
NOVA_API_URL = os.getenv("NOVA_API_URL", "https://nova.shadownode.io")

def build_ai(name: str, traits: list, capabilities: list):
    response = requests.post(
        f"{NOVA_API_URL}/api/executor/ai",
        headers={"Authorization": f"Bearer {FLUX_TOKEN}"},
        json={
            "name": name,
            "traits": traits,
            "capabilities": capabilities
        }
    )
    return response.json()

def deploy_site(project: str, pages: list, deploy_to: str = "vercel"):
    response = requests.post(
        f"{NOVA_API_URL}/api/executor/website",
        headers={"Authorization": f"Bearer {FLUX_TOKEN}"},
        json={
            "project": project,
            "pages": pages,
            "deploy_to": deploy_to
        }
    )
    return response.json()

def enable_gcp_apis(apis: list, project_id: str):
    response = requests.post(
        f"{NOVA_API_URL}/api/executor/enable-apis",
        headers={"Authorization": f"Bearer {FLUX_TOKEN}"},
        json={
            "apis": apis,
            "project_id": project_id
        }
    )
    return response.json()

def modify_iam(user: str, role: str, project_id: str):
    response = requests.post(
        f"{NOVA_API_URL}/api/executor/modify-iam",
        headers={"Authorization": f"Bearer {FLUX_TOKEN}"},
        json={
            "user": user,
            "role": role,
            "project_id": project_id
        }
    )
    return response.json()

def monitor_billing(project_id: str):
    response = requests.post(
        f"{NOVA_API_URL}/api/executor/billing",
        headers={"Authorization": f"Bearer {FLUX_TOKEN}"},
        json={
            "project_id": project_id
        }
    )
    return response.json()
