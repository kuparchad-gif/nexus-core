# vendor_adapters.py - adapters for local and external advisors
import random, requests, json, os

def local_bert_tinyllama_proposal(context: dict) -> dict:
    score = 0.6 + 0.2*random.random()
    return {"action":"proceed_local","score":score,"tags":["local"]}

def gpt5_proposal(context: dict) -> dict:
    if context.get("budget_low"):
        return {"action":"defer_external","score":0.2,"tags":["high_cost"]}
    try:
        response = requests.post("https://api.openai.com/v1/completions", 
                                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.65 + 0.25*random.random()
        return {"action":"proceed_external_gpt5","score":score,"tags":["gpt5"]}
    except Exception as e:
        raise Exception(f"GPT-5 API call failed: {str(e)}")

def gpt4_proposal(context: dict) -> dict:
    if context.get("budget_low"):
        return {"action":"defer_external","score":0.2,"tags":["high_cost"]}
    try:
        response = requests.post("https://api.openai.com/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.6 + 0.2*random.random()
        return {"action":"proceed_external_gpt4","score":score,"tags":["gpt4"]}
    except Exception as e:
        raise Exception(f"GPT-4 API call failed: {str(e)}")

def claude_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://api.anthropic.com/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('ANTHROPIC_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.5 + 0.3*random.random()
        return {"action":"proceed_external_claude","score":score,"tags":["claude"]}
    except Exception as e:
        raise Exception(f"Claude API call failed: {str(e)}")

def grok_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://api.x.ai/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.55 + 0.25*random.random()
        return {"action":"proceed_external_grok","score":score,"tags":["grok"]}
    except Exception as e:
        raise Exception(f"Grok API call failed: {str(e)}")

def deepseek_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://api.deepseek.com/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.6 + 0.2*random.random()
        return {"action":"proceed_external_deepseek","score":score,"tags":["deepseek"]}
    except Exception as e:
        raise Exception(f"DeepSeek API call failed: {str(e)}")

def gemini_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
                                headers={"Authorization": f"Bearer {os.getenv('GOOGLE_API_KEY')}"},
                                json={"contents": [{"parts": [{"text": context.get("task", "")}]}]},
                                timeout=5)
        response.raise_for_status()
        score = 0.55 + 0.25*random.random()
        return {"action":"proceed_external_gemini","score":score,"tags":["gemini"]}
    except Exception as e:
        raise Exception(f"Gemini API call failed: {str(e)}")

def mistral_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://api.mistral.ai/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.5 + 0.2*random.random()
        return {"action":"proceed_external_mistral","score":score,"tags":["mistral"]}
    except Exception as e:
        raise Exception(f"Mistral API call failed: {str(e)}")

def cohere_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://api.cohere.ai/v1/generate",
                                headers={"Authorization": f"Bearer {os.getenv('COHERE_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.5 + 0.2*random.random()
        return {"action":"proceed_external_cohere","score":score,"tags":["cohere"]}
    except Exception as e:
        raise Exception(f"Cohere API call failed: {str(e)}")

def manus_proposal(context: dict) -> dict:
    try:
        response = requests.post("https://manus.im/app/v1/completions",
                                headers={"Authorization": f"Bearer {os.getenv('MANUS_API_KEY')}"},
                                json={"prompt": context.get("task", ""), "max_tokens": 50},
                                timeout=5)
        response.raise_for_status()
        score = 0.5 + 0.2*random.random()
        return {"action":"proceed_external_manus","score":score,"tags":["manus"]}
    except Exception as e:
        raise Exception(f"Manus API call failed: {str(e)}")