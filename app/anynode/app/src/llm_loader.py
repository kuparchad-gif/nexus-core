# llm_loader.py
import json
import os
import sys
import requests
import time

# Constants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model_assignment.json")
LM_STUDIO_API = "http://localhost:1234/v1"

class LLMLoader:
    def __init__(self):
        self.model_assignments = self.load_model_assignments()
        self.active_models = {}
        
    def load_model_assignments(self):
        if not os.path.exists(MODEL_CONFIG_PATH):
            print(f"[LLM] Model assignment file not found: {MODEL_CONFIG_PATH}")
            return {}
            
        try:
            with open(MODEL_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LLM] Error loading model assignments: {e}")
            return {}
    
    def initialize_models(self):
        for role, model_info in self.model_assignments.items():
            model_name = model_info.get('model')
            if not model_name:
                continue
                
            print(f"[LLM] Initializing {model_name} for role: {role}")
            try:
                # Check if model is available in LM Studio
                response = requests.get(f"{LM_STUDIO_API}/models")
                if response.status_code == 200:
                    available_models = response.json()
                    if model_name in [m.get('id') for m in available_models.get('data', [])]:
                        self.active_models[role] = model_name
                        print(f"[LLM] Successfully loaded {model_name} for {role}")
                    else:
                        print(f"[LLM] Model {model_name} not available in LM Studio")
                else:
                    print(f"[LLM] Error checking available models: {response.status_code}")
            except Exception as e:
                print(f"[LLM] Error initializing model for {role}: {e}")
    
    def get_model_for_role(self, role):
        return self.active_models.get(role)

if __name__ == "__main__":
    print("[LLM] Starting LLM loader...")
    loader = LLMLoader()
    loader.initialize_models()
    print(f"[LLM] Active models: {loader.active_models}")
    
    # Keep the process running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[LLM] Shutting down LLM loader")