# At every node location:
from model_keys import load_model_key, validate_model_key

# Load any model instantly
llama_key = load_model_key("llama3.1_8b")
if validate_model_key(llama_key):
    print(f"✅ Loaded {llama_key['key']} - {llama_key['family']} architecture")
    
# Universal model factory
def create_model_from_key(key_data: dict):
    """Create model instance from universal key"""
    if key_data["family"] == "llama":
        return LlamaModel(key_data)
    elif key_data["family"] == "gemma":
        return GemmaModel(key_data)
    # ... all model families