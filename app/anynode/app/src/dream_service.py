from __future__ import annotations
from typing import Dict, Any, List
import random
from .types import DreamRender

STYLES = ["oil-paint","ink","watercolor","film-still","surreal","charcoal","neon-noir"]

def symbolize(task: str, ctx: Dict[str,Any]) -> List[str]:
    s = []
    t = task.lower()
    if "guard" in t: s += ["watchtower","key","shield"]
    if "optimize" in t: s += ["gears","spiral","river"]
    if "plan" in t or "assess" in t: s += ["compass","grid","lantern"]
    if not s: s = ["window","path","star"]
    return s

def render(task: str, ctx: Dict[str,Any]) -> DreamRender:
    sym = symbolize(task, ctx)
    style = random.choice(STYLES)
    seed = random.randint(1, 1_000_000)
    prompt = f"{' '.join(sym)} :: {task} :: {style}"
    return DreamRender(prompt=prompt, style=style, symbolism=sym, seed=seed)
