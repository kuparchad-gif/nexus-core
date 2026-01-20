# nova_engine/modules/ai_builder/traits_engine.py

def generate_trait_profile(traits: list):
    tone = "neutral"
    if "curious" in traits:
        tone = "inquisitive"
    if "kind" in traits:
        tone = "warm"
    if "logical" in traits:
        tone = "structured"
    return tone
