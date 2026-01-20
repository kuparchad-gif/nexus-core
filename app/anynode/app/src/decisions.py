from __future__ import annotations
from typing import List, Dict, Any, Optional

def labels_to_gbnf(labels: List[str]) -> str:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    alts = " | ".join(f'"{esc(l)}"' for l in labels)
    return f"root ::= choice\nchoice ::= {alts}\n"

def choose_one_with_llamacpp(llama, prompt: str, labels: List[str], max_tokens: int = 16) -> Dict[str, Any]:
    gbnf = labels_to_gbnf(labels)
    out = llama(prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0, grammar=gbnf, stop=["\n"])
    label = out["choices"][0]["text"].strip()
    return {"label": label, "scores": {l: (1.0 if l == label else 0.0) for l in labels}}

def choose_one_with_transformers(gen_fn, prompt: str, labels: List[str]) -> Dict[str, Any]:
    # gen_fn(prompt) -> str
    resp = gen_fn(prompt).strip()
    line = resp.splitlines()[0].strip().strip(" .,:;\"'").lower()
    for l in labels:
        if line == l.lower():
            return {"label": l, "scores": {x: (1.0 if x == l else 0.0) for x in labels}}
    for l in labels:  # substring fallback
        if l.lower() in line:
            return {"label": l, "scores": {x: (1.0 if x == l else 0.0) for x in labels}}
    return {"label": labels[0], "scores": {labels[0]: 1.0}}
