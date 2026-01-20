# Any node can now dynamically load any model
available_models = ["llama3.1_8b", "gemma2_1b", "code_llama_13b"]

for model_key in available_models:
    architecture = load_model_key(model_key)
    model_instance = create_model_from_key(architecture)
    # All models now speak the same interface