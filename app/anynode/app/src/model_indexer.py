# C:\Engineers\eden_engineering\scripts\utilities\model_indexer.py
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUT_FILE = os.path.join(os.path.dirname(__file__), 'models_available.txt')

model_ids = []
for root, dirs, files in os.walk(MODELS_DIR):
    for f in files:
        if f.endswith('.gguf') or f.endswith('.bin'):
            # Use filename minus extension as model id (adjust if you use different conventions)
            model_id = os.path.splitext(f)[0]
            model_ids.append(model_id)

with open(OUT_FILE, 'w', encoding='utf-8') as out:
    for mid in sorted(set(model_ids)):
        out.write(mid + '\n')

print(f"Indexed {len(model_ids)} models. See models_available.txt.")
