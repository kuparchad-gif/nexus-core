import re, asyncio, httpx
from typing import Dict, List, Tuple
from .config import EXPERTS, GATING_LM, LM_TOOLBOX_URL

KEYS = {
    "math": ["calculate", "sum", "integrate", "derivative", "arithmetic", "solve", "equation", "compute"],
    "graph": ["graph", "network", "nodes", "edges", "simulate", "topology", "response time", "erdos", "nexus"],
    "sentiment": ["sentiment", "positive", "negative", "tone", "emotion", "opinion", "review", "vibe"],
    "code": ["code", "python", "typescript", "bug", "error", "function", "class", "refactor", "complexity"],
    "llm": ["write", "explain", "summarize", "draft", "story", "analysis", "plan", "proposal"]
}

async def _score_with_llm(prompt: str) -> Dict[str, float]:
    # Optional LM ranker using LM Toolbox (adjust route/payload as needed)
    payload = {"prompt": f"Rank the best experts for: {prompt}\nExperts: {list(EXPERTS.keys())}\nReturn JSON mapping expert->score (0..1)."}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(f"{LM_TOOLBOX_URL}/v1/complete", json=payload)
            j = r.json()
            if isinstance(j, dict):
                return {k: float(v) for k, v in j.items() if k in EXPERTS}
    except Exception:
        pass
    return {}

def _heuristic_scores(prompt: str) -> Dict[str, float]:
    p = prompt.lower()
    scores = {name: 0.05 for name in EXPERTS.keys()}
    for name, words in KEYS.items():
        for w in words:
            if w in p:
                scores[name] += 0.2
    # numeric bias for math
    if re.search(r"\d+[\+\-\*/]", p):
        scores["math"] += 0.2
    # default llm nudge
    scores["llm"] += 0.1
    # normalize
    m = max(scores.values()) or 1.0
    return {k: v/m for k, v in scores.items()}

async def gate(prompt: str) -> Dict[str, float]:
    scores = _heuristic_scores(prompt)
    if GATING_LM:
        lm = await _score_with_llm(prompt)
        for k, v in lm.items():
            scores[k] = 0.5 * scores.get(k, 0.0) + 0.5 * min(max(v, 0.0), 1.0)
    return scores

async def call_expert(name: str, url: str, prompt: str) -> Tuple[str, dict]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{url}/infer", json={"prompt": prompt})
            return name, r.json()
    except Exception as e:
        return name, {"error": str(e)}

async def route(prompt: str, mode: str, k: int, combiner: str):
    scores = await gate(prompt)
    ranked = sorted([(n, scores[n], EXPERTS[n]) for n in scores], key=lambda x: x[1], reverse=True)

    selected: List[Tuple[str, float, str]]
    if mode == "best":
        selected = ranked[:1]
    elif mode == "broadcast":
        selected = ranked
    else:  # top_k
        selected = ranked[:max(1, k)]

    # parallel calls
    tasks = [call_expert(n, url, prompt) for (n, _, url) in selected]
    done = await asyncio.gather(*tasks)
    outputs = {name: payload for name, payload in done}

    # combine
    if combiner == "first":
        combined = next((payload for (_, payload) in done if "error" not in payload), outputs)
    elif combiner == "vote":
        # naive: if experts return {'label': 'positive'|'negative'|...} then majority vote
        labels = [p.get("label") for (_, p) in done if isinstance(p, dict) and p.get("label")]
        if labels:
            from collections import Counter
            combined = {"label": Counter(labels).most_common(1)[0][0], "votes": Counter(labels)}
        else:
            combined = max(outputs.values(), key=lambda v: len(json.dumps(v)) if isinstance(v, dict) else 0, default=outputs)
    else:
        # concat text field if exists
        parts = []
        for _, p in done:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        combined = {"text": "\n\n---\n\n".join(parts) if parts else str(outputs)}

    sel = [{"name": n, "url": url, "score": float(s)} for (n, s, url) in selected]
    return scores, sel, outputs, combined

