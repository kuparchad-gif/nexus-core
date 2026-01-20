from modal import Stub

deepseek = Stub.lookup("viren-trinity", "deepseek_coder")
gemma = Stub.lookup("viren-trinity", "gemma_small")
tinyllama = Stub.lookup("viren-trinity", "tinyllama_sql")
mistral = Stub.lookup("viren-trinity", "mistral_mini")

def route_and_merge(prompt: str):
    logs = []
    result = ""

    if "sql" in prompt.lower():
        a = tinyllama.call(prompt)
        logs.append(f"🧠 Routed to TinyLLaMA for SQL parsing.")
        result += a
    elif "debug" in prompt or "fix" in prompt:
        a = deepseek.call(prompt)
        b = gemma.call(prompt)
        logs.append("🧠 Dual route: Deepseek (debug) + Gemma (narrative).")
        result += f"\n{a}\n{b}"
    else:
        a = gemma.call(prompt)
        b = mistral.call(prompt)
        logs.append("🧠 Gemma & Mistral handled general reasoning.")
        result += f"\n{a}\n{b}"

    return "\n".join(logs), result
