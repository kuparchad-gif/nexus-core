import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIEngine:
    def __init__(self, registry_path="Config/apis/api_registry.json"):
        self.registry_path = registry_path
        self.apis = self.load_registry()

    def load_registry(self):
        """Load all available APIs from the registry."""
        try:
            with open(self.registry_path, "r") as file:
                data = json.load(file)
                return data.get("apis", [])
        except Exception as e:
            print(f"❗ Error loading API registry: {e}")
            return []

    def list_apis(self):
        """List all available APIs."""
        return [{"name": api["name"], "type": api["type"]} for api in self.apis]

    def get_api_info(self, name):
        """Retrieve details of a specific API by name."""
        for api in self.apis:
            if api["name"].lower() == name.lower():
                return api
        return None

    def call_api(self, name, endpoint, params=None, method="GET"):
        """Make a request to an API."""
        api_info = self.get_api_info(name)
        if not api_info:
            return {"error": "API not found."}

        headers = {}
        if api_info["auth_required"]:
            token = os.getenv(api_info["token_env_var"])
            if not token:
                return {"error": "API key missing in environment variables."}
            headers["Authorization"] = f"Bearer {token}"

        full_url = f"{api_info['base_url']}/{endpoint.lstrip('/')}"

        try:
            if method == "GET":
                response = requests.get(full_url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(full_url, headers=headers, json=params)
            else:
                return {"error": "Unsupported HTTP method."}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

