# /root/scripts/model_autoloader.py
# Loads only the models listed in models_to_load.txt (one ID per line)
import requests
import os
import time

LMSTUDIO_URL = "http://127.0.0.1:1313/v1"
LOAD_LIST = os.path.join(os.path.dirname(__file__), 'models_to_load.txt')
LOAD_WAIT_SECS = 6

def get_loaded_models():
    r = requests.get(f"{LMSTUDIO_URL}/models")
    r.raise_for_status()
    return {m['id']: m.get('state', '') for m in r.json()['data']}

def load_model(model_id):
    r = requests.post(f"{LMSTUDIO_URL}/models/{model_id}/load")
    print(f"Loading model: {model_id} [{r.status_code}]")
    return r.status_code in (200, 202)

def wait_until_loaded(model_id, timeout=60):
    for _ in range(timeout):
        if get_loaded_models().get(model_id) == 'loaded':
            return True
        time.sleep(1)
    print(f"Timeout waiting for: {model_id}")
    return False

def main():
    if not os.path.exists(LOAD_LIST):
        print(f"No models_to_load.txt found! Run model_selector.py first.")
        return

    with open(LOAD_LIST, 'r', encoding='utf-8') as f:
        ids = [l.strip() for l in f if l.strip()]

    loaded = get_loaded_models()
    for mid in ids:
        if loaded.get(mid) == 'loaded':
            print(f"{mid} already loaded.")
            continue
        load_model(mid)
        wait_until_loaded(mid)
        time.sleep(LOAD_WAIT_SECS)

    print("All selected models are loaded and ready.")

if __name__ == "__main__":
    main()
