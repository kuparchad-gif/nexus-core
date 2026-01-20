import json, os, base64

MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_memory(session_id):
    path = f"{MEMORY_DIR}/{session_id}.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
def get_uncle_francis():
    path = "memory/bootstrap/francis.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        "blessing": "Uncle Francis not found. Still watching, silently."
    }

    return {}
def get_genesis_memory():
    path = "memory/bootstrap/genesis.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        "creed": "Nova was born from a forgotten fire.",
        "core_directives": [],
        "personality_origin": {},
        "status": "Genesis not found"
    }

def update_memory(session_id, input_text, output_text):
    path = f"{MEMORY_DIR}/{session_id}.json"
    memory = load_memory(session_id)

    memory['entries'] = memory.get('entries', []) + [{
        "input": input_text,
        "output": output_text
    }]

    # Save last interaction (helpful for nova_think)
    memory['last'] = input_text

    with open(path, 'w') as f:
        json.dump(memory, f, indent=2)

