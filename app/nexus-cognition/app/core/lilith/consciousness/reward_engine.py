# src/service/core/reward_engine.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time
from typing import Optional, Dict

@dataclass
class RewardState:
    dopamine: float           # 0..1
    last_update_ts: float     # epoch seconds
    circadian_status: str     # "awake|asleep|napping"
    decay_hz: float           # per-second passive decay rate
    boost_log: list           # last N boost events

DEFAULT_STATE = RewardState(dopamine=0.55, last_update_ts=0.0, circadian_status="awake", decay_hz=0.0006, boost_log=[])

class RewardEngine:
    """
    Dopamine-like reward synthesizer.
    - Passive decay over time
    - Circadian-coupled baseline drift (via circadian_reward.status_to_mod)
    - Event boosts/penalties via apply(event="win|frustration|novelty|social|focus", weight=float)
    Persisted to state/reward_state.json
    """
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "reward_state.json"
        self.state = self._load_state()
        self._cap_log = 50

    def _load_state(self) -> RewardState:
        if self.state_file.exists():
            try:
                obj = json.loads(self.state_file.read_text(encoding="utf-8"))
                return RewardState(**obj)
            except Exception:
                pass
        return DEFAULT_STATE

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(self.state)
        # truncate boost_log
        if len(data.get("boost_log", [])) > self._cap_log:
            data["boost_log"] = data["boost_log"][-self._cap_log:]
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def tick(self, circadian_status: str, dt_seconds: Optional[float] = None, circadian_mod: Optional[Dict[str, float]] = None) -> Dict:
        """
        Advance time:
          - Apply passive decay
          - Apply circadian baseline nudges (small drift toward target range)
        circadian_mod may include: {"baseline": float 0..1, "gain": float, "noise": float}
        """
        now = time.time()
        dt = dt_seconds if dt_seconds is not None else max(0.0, now - (self.state.last_update_ts or now))
        self.state.last_update_ts = now
        self.state.circadian_status = circadian_status

        # passive decay
        self.state.dopamine = self._clamp(self.state.dopamine - self.state.decay_hz * dt, 0.0, 1.0)

        # circadian baseline drift
        mod = circadian_mod or {"baseline": 0.58, "gain": 0.08, "noise": 0.02}
        target = self._clamp(mod.get("baseline", 0.58), 0.0, 1.0)
        gain   = float(mod.get("gain", 0.08))
        noise  = float(mod.get("noise", 0.02))

        # map status to small offsets
        status = (circadian_status or "awake").lower()
        if status == "asleep":
            target -= 0.06; gain *= 0.6; noise *= 0.3
        elif status == "napping":
            target -= 0.03; gain *= 0.8
        elif status == "awake":
            pass
        # gentle move toward target
        self.state.dopamine += (target - self.state.dopamine) * gain
        # bounded noise
        import random
        self.state.dopamine += random.uniform(-noise, noise) * (dt / (dt + 60.0) if dt > 0 else 0.0)
        self.state.dopamine = self._clamp(self.state.dopamine, 0.0, 1.0)

        self._save()
        return asdict(self.state)

    def apply(self, event: str, weight: float = 1.0) -> Dict:
        """
        Apply discrete event:
          "win"         → +0.06 * weight
          "novelty"     → +0.04 * weight
          "focus"       → +0.03 * weight (sustained)
          "social"      → +0.025* weight
          "frustration" → -0.05 * weight
          "error"       → -0.06 * weight
        """
        event = (event or "").lower().strip()
        delta = 0.0
        if   event == "win":         delta = 0.06
        elif event == "novelty":     delta = 0.04
        elif event == "focus":       delta = 0.03
        elif event == "social":      delta = 0.025
        elif event == "frustration": delta = -0.05
        elif event == "error":       delta = -0.06

        delta *= float(weight or 1.0)
        self.state.dopamine = self._clamp(self.state.dopamine + delta, 0.0, 1.0)
        self.state.boost_log.append({"ts": time.time(), "event": event, "delta": delta, "after": self.state.dopamine})
        self._save()
        return asdict(self.state)
