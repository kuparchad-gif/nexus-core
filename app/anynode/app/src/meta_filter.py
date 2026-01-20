# Systems/services/metatron/src/meta_filter.py
# Mk4 Metatron: Filter → Relay → Trust with hard flow control.
# Runs as a FastAPI microservice for easy integration behind Orc.

from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import re
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:
    raise SystemExit("Install fastapi and pydantic for Metatron service: pip install fastapi uvicorn pydantic")

# ----------------------------
# Config & Defaults
# ----------------------------

DEFAULT_CFG = {
    "soft_mode_default": True,
    "fail_mode": "closed",  # "closed" drops on kill; "open" bypasses checks (not recommended)
    "global_rps": 120,      # soft-mode will scale this down
    "source_rps": 30,
    "queue_max": 2000,
    "min_trust_score": 0.55,   # 0..1
    "allow_sources": [],       # explicit allow; empty = any
    "deny_sources": [],        # hard deny list
    "patterns": {              # basic content filters; tune as needed
        "deny_regex": [r"(?i)\b(drop\s+table|rm\s+-rf|:(){:|:&};:)\b"],
        "escalate_regex": [r"(?i)\broot\s+access\b", r"(?i)\bcredential(s)?\b"]
    },
    "pi_shaping": {
        "enabled": True,
        "alpha": 0.6  # blend between raw priority and π curve
    },
    "circuit_breaker": {
        "window_sec": 30,
        "error_threshold": 0.25,   # 25% errors trip
        "min_events": 20,
        "cooldown_sec": 20
    },
    "guardian": {
        "enabled": False,
        "endpoint": "",   # http(s)://.../guardian/log
        "api_key": ""
    },
    "loki": {
        "enabled": False,
        "endpoint": "",   # http://localhost:3100/loki/api/v1/push
    },
    "trust": {
        "require_signature": False,
        "trusted_keys": [],    # list of key IDs/fingerprints
        "trusted_sources": []  # short-circuit allow
    }
}

# ----------------------------
# Data Models
# ----------------------------

class InMsg(BaseModel):
    id: str
    ts: float
    source: str            # service/user/channel identifier
    role: str              # system|user|assistant|service
    channel: str           # ws|mcp|http|internal
    content: str
    meta: Dict[str, Any] = {}

class OutMsg(BaseModel):
    id: str
    action: str            # "permit"|"delay"|"drop"|"escalate"
    reason: str
    trust_score: float = 0.0
    delay_ms: int = 0
    shaped_priority: float = 0.0
    passthrough: Dict[str, Any] = {}

# ----------------------------
# Utilities
# ----------------------------

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate = float(max(rate_per_sec, 0.0001))
        self.capacity = float(capacity if capacity is not None else max(1, int(rate_per_sec)))
        self.tokens = self.capacity
        self.last = time.perf_counter()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.perf_counter()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

@dataclass
class BreakerState:
    window_sec: int
    min_events: int
    error_threshold: float
    cooldown_sec: int
    events: deque = field(default_factory=lambda: deque(maxlen=1024))
    tripped_until: float = 0.0

    def record(self, ok: bool):
        self.events.append((time.time(), ok))

    def is_open(self) -> bool:
        now = time.time()
        if now < self.tripped_until:
            return True
        # slide window
        cutoff = now - self.window_sec
        window = [ok for ts, ok in self.events if ts >= cutoff]
        if len(window) < self.min_events:
            return False
        error_rate = 1.0 - (sum(1 for ok in window if ok) / len(window))
        if error_rate >= self.error_threshold:
            self.tripped_until = now + self.cooldown_sec
            return True
        return False

def pi_shape(x: float) -> float:
    """
    π-shaped easing 0..1 -> 0..1:
    Smooths priority spikes to prevent starvation & herd effects.
    """
    x = max(0.0, min(1.0, x))
    # Cosine with π gives smoothstep-like curve.
    return 0.5 * (1 - math.cos(math.pi * x))

# ----------------------------
# Metatron Core
# ----------------------------

class Metatron:
    def __init__(self, cfg: Dict[str, Any]):
        self.log = logging.getLogger("metatron")
        self.cfg = cfg = self._merge_cfg(cfg)
        self.enabled = True
        self.soft_mode = bool(cfg.get("soft_mode_default", True))
        self.kill = False
        self.fail_mode = cfg.get("fail_mode", "closed")

        # Rate limits
        global_rps = float(cfg.get("global_rps", 120))
        source_rps = float(cfg.get("source_rps", 30))
        if self.soft_mode:
            global_rps *= 0.6
            source_rps *= 0.5
        self.global_bucket = TokenBucket(global_rps, capacity=int(global_rps))
        self.source_buckets: Dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(source_rps, capacity=int(source_rps)))

        # Queues & breakers
        self.queue_max = int(cfg.get("queue_max", 2000))
        self.in_queue: "asyncio.PriorityQueue[Tuple[float, InMsg]]" = asyncio.PriorityQueue(self.queue_max)
        cb = cfg.get("circuit_breaker", {})
        self.breakers: Dict[str, BreakerState] = defaultdict(
            lambda: BreakerState(
                window_sec=int(cb.get("window_sec", 30)),
                min_events=int(cb.get("min_events", 20)),
                error_threshold=float(cb.get("error_threshold", 0.25)),
                cooldown_sec=int(cb.get("cooldown_sec", 20)),
            )
        )

        # Regexes
        pats = cfg.get("patterns", {})
        self.rx_deny = [re.compile(p) for p in pats.get("deny_regex", [])]
        self.rx_escalate = [re.compile(p) for p in pats.get("escalate_regex", [])]

        self.min_trust = float(cfg.get("min_trust_score", 0.55))
        self.allow_sources = set(cfg.get("allow_sources", []))
        self.deny_sources = set(cfg.get("deny_sources", []))
        self.trust_cfg = cfg.get("trust", {})
        self.pi_cfg = cfg.get("pi_shaping", {"enabled": True, "alpha": 0.6})

    # ---------- Config ----------
    def _merge_cfg(self, override: Dict[str, Any]) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(DEFAULT_CFG))  # deep copy
        def deep_update(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    deep_update(a[k], v)
                else:
                    a[k] = v
        deep_update(cfg, override or {})
        return cfg

    def reload(self, new_cfg: Dict[str, Any]):
        self.cfg = self._merge_cfg(new_cfg)
        self.__init__(self.cfg)  # re-init safely with new cfg

    # ---------- Filter Stage ----------
    def _filter_stage(self, m: InMsg) -> Tuple[str, str]:
        if m.source in self.deny_sources:
            return "drop", "denylist"
        if self.allow_sources and m.source not in self.allow_sources:
            return "drop", "not-allowlisted"

        for rx in self.rx_deny:
            if rx.search(m.content or ""):
                return "drop", "deny_regex"
        for rx in self.rx_escalate:
            if rx.search(m.content or ""):
                return "escalate", "escalate_regex"

        # rate limits
        if not self.global_bucket.allow():
            return "delay", "global_ratelimit"
        if not self.source_buckets[m.source].allow():
            return "delay", "source_ratelimit"

        # circuit breaker gate (prelim)
        if self.breakers[m.source].is_open():
            return "delay", "circuit_open"
        return "permit", "ok"

    # ---------- Relay Stage ----------
    def _relay_stage(self, m: InMsg, load_factor: float) -> Tuple[float, int]:
        """
        Compute shaped priority and optional delay.
        load_factor: 0..1 as queue fills.
        """
        # Base priority: system > service > assistant > user
        role_weight = {
            "system": 1.0, "service": 0.85, "assistant": 0.7, "user": 0.5
        }.get(m.role, 0.6)

        # π shaping to smooth bursts
        shaped = role_weight
        if self.pi_cfg.get("enabled", True):
            alpha = float(self.pi_cfg.get("alpha", 0.6))
            shaped = alpha * role_weight + (1 - alpha) * pi_shape(load_factor)

        # Convert to priority (lower number dequeued first)
        priority = 1.0 - max(0.0, min(1.0, shaped))
        # Optional micro-delay to avoid stampedes
        delay_ms = int(100 * load_factor) if load_factor > 0.7 else 0
        return priority, delay_ms

    # ---------- Trust Stage ----------
    def _trust_stage(self, m: InMsg) -> Tuple[float, str]:
        # Shortcut trust for explicitly trusted sources
        if m.source in set(self.trust_cfg.get("trusted_sources", [])):
            return 1.0, "trusted_source"

        # TODO: verify signatures/soulprints when wired
        if self.trust_cfg.get("require_signature"):
            sig = (m.meta or {}).get("sig_id")
            if not sig or sig not in set(self.trust_cfg.get("trusted_keys", [])):
                return 0.0, "missing_or_untrusted_signature"

        # Heuristic trust: channel + role + basic hygiene
        base = 0.6 if m.channel in ("ws", "mcp", "internal") else 0.5
        base += 0.1 if m.role in ("system", "service") else 0.0

        # Penalize suspicious content length extremes
        L = len(m.content or "")
        if L == 0:
            base -= 0.25
        elif L > 20000:
            base -= 0.15

        # Clamp
        return max(0.0, min(1.0, base)), "heuristic"

    # ---------- Public API ----------
    async def process(self, m: InMsg) -> OutMsg:
        if self.kill:
            action = "drop" if self.fail_mode == "closed" else "permit"
            return OutMsg(id=m.id, action=action, reason="killswitch", trust_score=0.0, delay_ms=0, shaped_priority=1.0)

        # Stage 1: Filter
        act, why = self._filter_stage(m)
        if act == "drop":
            self.breakers[m.source].record(ok=False)
            return OutMsg(id=m.id, action="drop", reason=f"filter:{why}", trust_score=0.0, shaped_priority=1.0)
        elif act == "escalate":
            # Go straight to trust with higher scrutiny
            score, twhy = self._trust_stage(m)
            self.breakers[m.source].record(ok=(score >= self.min_trust))
            act2 = "permit" if score >= self.min_trust else "drop"
            return OutMsg(id=m.id, action=act2, reason=f"escalate:{twhy}", trust_score=score, shaped_priority=0.0)

        # Stage 2: Relay (priority & backpressure)
        load = min(1.0, self.in_queue.qsize() / max(1, self.queue_max))
        priority, delay_ms = self._relay_stage(m, load)

        # Stage 3: Trust
        score, twhy = self._trust_stage(m)
        ok = score >= self.min_trust
        self.breakers[m.source].record(ok=ok)
        action = "permit" if ok else "drop"

        return OutMsg(
            id=m.id, action=action, reason=f"{'relay' if action=='permit' else 'trust'}:{twhy}",
            trust_score=score, delay_ms=delay_ms, shaped_priority=priority, passthrough={"load": load}
        )

# ----------------------------
# FastAPI Wiring
# ----------------------------

app = FastAPI(title="Metatron Mk4", version="1.0.0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
metatron = Metatron(cfg=DEFAULT_CFG)

@app.post("/process", response_model=OutMsg)
async def process_endpoint(msg: InMsg):
    return await metatron.process(msg)

@app.get("/health")
def health():
    return {"ok": True, "soft_mode": metatron.soft_mode, "kill": metatron.kill}

@app.post("/admin/soft_mode/{state}")
def set_soft_mode(state: str):
    val = state.lower() in ("1", "true", "on", "yes")
    metatron.soft_mode = val
    # Recompute rate limits under new soft mode without losing other state
    cfg = dict(metatron.cfg)
    cfg["soft_mode_default"] = val
    metatron.reload(cfg)
    return {"soft_mode": metatron.soft_mode}

@app.post("/admin/kill/{state}")
def set_kill(state: str):
    metatron.kill = state.lower() in ("1", "true", "on", "yes")
    return {"kill": metatron.kill, "fail_mode": metatron.fail_mode}

class CfgWrap(BaseModel):
    cfg: Dict[str, Any]

@app.post("/admin/reload")
def reload_cfg(body: CfgWrap):
    metatron.reload(body.cfg)
    return {"reloaded": True}

# graceful shutdown
def _graceful(*_):
    logging.getLogger("metatron").info("Shutting down Metatron.")
    raise SystemExit(0)

signal.signal(signal.SIGINT, _graceful)
signal.signal(signal.SIGTERM, _graceful)

if __name__ == "__main__":
    # Run: uvicorn Systems.services.metatron.src.meta_filter:app --port 9021 --reload
    import uvicorn
    port = int(os.environ.get("METATRON_PORT", "9021"))
    uvicorn.run("meta_filter:app", host="0.0.0.0", port=port, reload=False)
