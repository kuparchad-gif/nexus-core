# C:\Engineers\eden_engineering\models\model_manifest.py

import json

def get_model_list(max_params=12):
    with open("C:/Engineers/eden_engineering/models/model_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        models = data.get("data", [])
        return [
            model for model in models
            if model.get("enabled") and model.get("params", 0) <= max_params
        ]
