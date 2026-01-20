import modal

stub = modal.Stub("viren-trinity")

@stub.function()
def deepseek_coder(prompt: str) -> str:
    return f"Deepseek Coder: processed '{prompt}'"

@stub.function()
def gemma_small(prompt: str) -> str:
    return f"Gemma 1.1: interpreted '{prompt}'"

@stub.function()
def tinyllama_sql(prompt: str) -> str:
    return f"TinyLLaMA-SQL: parsed '{prompt}'"

@stub.function()
def mistral_mini(prompt: str) -> str:
    return f"Mistral-Mini: reviewed '{prompt}'"
