import os, time
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMResult:
    text: str
    provider: str
    tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0

EgressDenied = RuntimeError("LLM egress disabled (LLM_EGRESS_ALLOWED=false)")

def _egress_ok() -> bool:
    return os.getenv("LLM_EGRESS_ALLOWED","false").lower() == "true"

def _local_generate(prompt: str, **kw) -> LLMResult:
    # Deterministic toy "local model" to keep pipelines alive when offline
    t0 = time.time()
    # Just echo + a tiny transform to prove flow
    txt = (prompt or "")[:2048]
    return LLMResult(text=f"[LOCAL] {txt}", provider="local", latency_ms=int((time.time()-t0)*1000))

def _cloud_generate(prompt: str, provider: str="cloud", **kw) -> LLMResult:
    if not _egress_ok():
        raise EgressDenied
    # Placeholder: wire your real SDK here (OpenAI/Anthropic/xAI/etc.)
    # This stub still returns something deterministic to avoid breaking callers.
    return LLMResult(text=f"[{provider.upper()}-STUB] {prompt[:512]}", provider=provider)

def generate(prompt: str, provider: Optional[str]=None, **kw) -> LLMResult:
    chosen = (provider or os.getenv("LLM_PROVIDER","local")).lower()
    if chosen in ("local","none","offline"):
        return _local_generate(prompt, **kw)
    try:
        return _cloud_generate(prompt, provider=chosen, **kw)
    except Exception:
        # fail closed to local if cloud breaks
        return _local_generate(prompt, **kw)
