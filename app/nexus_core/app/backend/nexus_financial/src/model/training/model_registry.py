import requests

class ModelRegistry:
    def __init__(self, systems_config):
        self.systems_config = systems_config
        self.available_models = []

    def discover_models(self):
        """Discover available models from all configured systems"""
        print("Discovering models from configured systems...")
        
        for system_name, config in self.systems_config.items():
            print(f"Checking system: {system_name} at {config['url']}")
            
            if config["type"] == "openai_compatible":
                self._discover_openai_compatible_models(system_name, config["url"])
            elif config["type"] == "ollama":
                self._discover_ollama_models(system_name, config["url"])
            else:
                print(f"Unknown system type: {config['type']} for system {system_name}")
                
        return self.available_models

    def _discover_openai_compatible_models(self, system_name, url):
        """Discover models from OpenAI-compatible API endpoints"""
        try:
            response = requests.get(f"{url}/models", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            model_count = len(models_data.get("data", []))
            
            for model in models_data.get("data", []):
                self.available_models.append({
                    "id": model["id"],
                    "system": system_name,
                    "type": "openai_compatible",
                    "url": url
                })
                
            print(f"Discovered {model_count} models from {system_name}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error discovering models from {system_name} (OpenAI compatible): {e}")

    def _discover_ollama_models(self, system_name, url):
        """Discover models from Ollama API endpoints"""
        try:
            response = requests.get(f"{url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            model_count = len(models_data.get("models", []))
            
            for model in models_data.get("models", []):
                self.available_models.append({
                    "id": model["name"],
                    "system": system_name,
                    "type": "ollama",
                    "url": url
                })
                
            print(f"Discovered {model_count} models from {system_name}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error discovering models from {system_name} (Ollama): {e}")

    def get_models_by_system(self, system_name):
        """Get all models from a specific system"""
        return [model for model in self.available_models if model["system"] == system_name]

    def get_model_by_id(self, model_id):
        """Get a specific model by ID"""
        for model in self.available_models:
            if model["id"] == model_id:
                return model
        return None

    def clear_models(self):
        """Clear the current model registry"""
        self.available_models = []