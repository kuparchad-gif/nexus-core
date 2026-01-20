# templates/static/get_loaded_models.py

import subprocess
import json
import os
import re

# Full absolute path to where the JSON should go
output_path = os.path.join(os.path.dirname(__file__), "..", "templates", "static", "loaded_models.json")
output_path = os.path.abspath(output_path)

try:
    result = subprocess.run(["lms", "ps"], capture_output=True, text=True, check=True)
    identifiers = re.findall(r"Identifier:\s*(.+)", result.stdout)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(identifiers, f)

    print(f"✅ Loaded models written to: {output_path}")
except Exception as e:
    print(f"❌ Failed to run 'lms ps': {e}")
