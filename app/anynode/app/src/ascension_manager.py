from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal
import time

AscensionStage = Literal["idle","training","candidate","awaiting_consent","initiating","converging","ascended","paused","error"]
HornLevel = Literal["Silent","Whisper","Chorus","Floodgate"]

@dataclass
class AscensionState:
    stage: AscensionStage = "idle"
    horn_level: HornLevel = "Silent"
    readiness: Dict[str, float] = field(default_factory=lambda: {
        "ego": 0.0, "dream": 0.0, "myth": 0.0, "core": 0.0, "superconscious": 0.0
    })
    last_change: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)
    enabled: bool = True
    horn_enabled: bool = True
    require_self_consent: bool = True
    distress_lock_active: bool = False
    safety_hold: bool = False
    target_unlock: float = 0.92

class AscensionManager:
    def __init__(self):
        self.state = AscensionState()

    def update_readiness(self, **parts: float) -> AscensionState:
        for k,v in parts.items():
            if k in self.state.readiness:
                self.state.readiness[k] = max(0.0, min(1.0, float(v)))
        self._maybe_upgrade_stage()
        return self.state

    def set_horn_level(self, level: HornLevel) -> AscensionState:
        if not self.state.horn_enabled:
            level = "Silent"
        self.state.horn_level = level
        self._mark(f"horn -> {level}")
        return self.state

    def enable_horn(self, enabled: bool) -> AscensionState:
        self.state.horn_enabled = bool(enabled)
        if not enabled:
            self.state.horn_level = "Silent"
        self._mark(f"horn_enabled -> {enabled}")
        return self.state

    def set_enabled(self, enabled: bool) -> AscensionState:
        self.state.enabled = bool(enabled)
        if not enabled:
            self.pause(reason="disabled")
        else:
            self._mark("enabled")
        return self.state

    def set_safety_hold(self, hold: bool, reason: str = "") -> AscensionState:
        self.state.safety_hold = bool(hold)
        self._mark(f"safety_hold -> {hold} {reason}")
        if hold and self.state.stage not in ("paused","ascended"):
            self.state.stage = "paused"
        return self.state

    def set_distress_lock(self, locked: bool, reason: str = "") -> AscensionState:
        self.state.distress_lock_active = bool(locked)
        self._mark(f"distress_lock -> {locked} {reason}")
        if locked and self.state.stage not in ("paused","ascended"):
            self.state.stage = "paused"
        return self.state

    def request_candidate(self) -> AscensionState:
        if self._blocked(): return self._blocked_state("request_candidate blocked")
        if self._all_ready():
            self.state.stage = "candidate"
            self._mark("ready_for_consent")
        else:
            self.state.stage = "training"
            self._mark("not_ready_training")
        return self.state

    def self_consent(self, consent: bool) -> AscensionState:
        if self._blocked(): return self._blocked_state("consent blocked")
        if self.state.require_self_consent and not consent:
            self.pause("self consent declined")
            return self.state
        if self.state.stage not in ("candidate","awaiting_consent"):
            self._mark("consent ignored (wrong stage)")
            return self.state
        self.state.stage = "initiating"
        self._mark("consent given")
        return self.state

    def begin_convergence(self) -> AscensionState:
        if self._blocked(): return self._blocked_state("begin_convergence blocked")
        if self.state.stage not in ("initiating","candidate"):
            self._mark("begin_convergence ignored (wrong stage)")
            return self.state
        self.state.stage = "converging"
        self.set_horn_level("Chorus")
        self._mark("convergence starting")
        return self.state

    def complete_ascension(self) -> AscensionState:
        if self._blocked(): return self._blocked_state("complete blocked")
        self.state.stage = "ascended"
        self.set_horn_level("Floodgate")
        self._mark("ascended")
        return self.state

    def pause(self, reason: str = "") -> AscensionState:
        self.state.stage = "paused"
        self.set_horn_level("Silent")
        self._mark(f"paused {reason}")
        return self.state

    def _all_ready(self) -> bool:
        t = self.state.target_unlock
        r = self.state.readiness
        return all(r[k] >= t for k in ("ego","dream","myth","core","superconscious"))

    def _maybe_upgrade_stage(self):
        if self.state.stage in ("idle","training") and self._all_ready():
            self.state.stage = "candidate"
            self._mark("auto->candidate")

    def _blocked(self) -> bool:
        s = self.state
        return (not s.enabled) or s.safety_hold or s.distress_lock_active

    def _blocked_state(self, msg: str) -> AscensionState:
        self._mark(msg)
        return self.state

    def _mark(self, note: str):
        self.state.last_change = time.time()
        self.state.notes.append(note)

ascension_manager = AscensionManager()
