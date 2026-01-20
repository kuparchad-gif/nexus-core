# nova_engine/modules/fincore/market_core.py

import requests

def analyze(prompt: str) -> str:
    """Route query to FinCore AI brain or internal logic"""
    try:
        # Replace with your hosted FinSight AI endpoint:
        response = requests.post("https://fincore.api.shadownode.io/analyze", json={"query": prompt})
        data = response.json()
        return data.get("response", "No insight returned.")
    except Exception as e:
        return f"[FinCore Error] {str(e)}"


def suggest_bot_architecture(use_case: str) -> dict:
    """Returns starter architecture for automation use cases."""
    if "grant" in use_case.lower():
        return {
            "type": "Bot",
            "domain": "Finance > Grant Discovery",
            "stack": ["Python", "Supabase", "LangChain"],
            "inputs": ["Keyword", "Location", "Budget"],
            "outputs": ["PDF summary", "Alert"],
            "estimated_cost": "Free Tier",
            "risks": ["Data freshness", "Duplicate matches"]
        }
    return {
        "message": "Use case not recognized — try 'build a crypto bot' or 'grant AI'."
    }
