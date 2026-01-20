import modal

stub = modal.Stub("viren-demo")

@stub.function()
def route_prompt(prompt: str) -> str:
    if "sql" in prompt:
        return "TinyLLaMA will handle SQL."
    elif "debug" in prompt or "fix" in prompt:
        return "Deepseek will debug your issue."
    else:
        return "Gemma is responding with a general answer."
