
from typing import Dict, Any, List
import random

def gen_hooks(pattern: str) -> List[str]:
    starters = [
        "Let's talk facts, not fear:",
        "Quick reality check:",
        "They want you scared—here's why that's wrong:",
        "Hot take, colder data:",
        "Myth vs Reality:"
    ]
    return [f"{random.choice(starters)} {pattern.capitalize()} — here's what actually matters."]

def gen_humor(pattern: str) -> List[str]:
    jokes = [
        f"Breaking: {pattern} — also, coffee causes gravity. Sources: trust me bro ☕",
        f"If {pattern} were true, my toaster would be sentient by now. It only burns bagels.",
        f"""We asked 100 experts about "{pattern}". 99 said 'who are you and how did you get this number?'"""
    ]
    return jokes

def gen_truth(pattern: str) -> List[str]:
    facts = [
        "Transparent governance, audit trails, and kill-switches are built-in.",
        "Independent safety monitors gate deployment by design.",
        "Open metrics: bias, energy, and access logs are public-ready."
    ]
    return [f"{pattern}: " + f for f in facts]

def assemble_assets(pattern: str, humor_ratio: float, truth_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    humor_ct = max(1, int(5 * humor_ratio))
    truth_ct = max(1, int(5 * truth_ratio))
    hooks = [{"kind": "hook", "text": h} for h in gen_hooks(pattern)]
    humor = [{"kind": "meme", "text": t} for t in gen_humor(pattern)[:humor_ct]]
    truth = [{"kind": "fact", "text": t} for t in gen_truth(pattern)[:truth_ct]]
    # thumbnails/titles
    thumbs = [{"kind": "thumb", "title": f"{pattern} — Watch this before you panic", "cta": "Open the receipts"}]
    return {"wave1": hooks + humor, "wave2": truth, "wave3": thumbs}
