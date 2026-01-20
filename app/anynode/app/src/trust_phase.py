# src/lilith/policy/trust_phase.py
import json, os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

PHASES_TOTAL = 30  # 30 years → 30 phases (drop 1 level per year)

def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _generate_phases(seed_start_iso: str) -> Dict[str, Any]:
    start = _iso_to_dt(seed_start_iso)
    phases = []
    # Policy level starts at 30 and drops to 1 across 30 years
    for i in range(PHASES_TOTAL):
        phase_number = i + 1                      # 1..30
        policy_level = max(PHASES_TOTAL - i, 1)   # 30..1
        phases.append({
            "phase": phase_number,
            "policy_level": policy_level,
            "year_offset": i,
            "effective_from": (start.replace(year=start.year + i)).isoformat()
        })
    return {"trust_start_date": seed_start_iso, "phases": phases}

@dataclass
class TrustState:
    active_phase: int
    policy_level: int

class TrustPhaseMixin:
    """
    Env:
      LILITH_SOUL_SEED=seeds\\lilith_soul_seed.migrated.json
      LILITH_TRUST_PHASES=seeds\\trust_phases.json
    Gates (min levels):
      can_autopatch       <= 20
      can_verbose_logs    <= 15
      can_open_tools      <= 10
      can_full_autonomy   <= 1
    """

    _seed_path = Path(os.getenv("LILITH_SOUL_SEED", "seeds/lilith_soul_seed.migrated.json"))
    _phases_path = Path(os.getenv("LILITH_TRUST_PHASES", "seeds/trust_phases.json"))

    def __init__(self) -> None:
        self._trust_state = self._compute_trust_state()

    # ---- Public API ----
    def policy_dict(self) -> Dict[str, Any]:
        st = self._trust_state
        return {
            "active_phase": st.active_phase,
            "policy_level": st.policy_level,
            "gates": self.gates()
        }

    def gate(self, min_level: int) -> bool:
        """Allow action when current policy_level <= min_level."""
        return self._trust_state.policy_level <= int(min_level)

    def gates(self) -> Dict[str, bool]:
        lvl = self._trust_state.policy_level
        return {
            "can_autopatch":      (lvl <= 20),
            "can_verbose_logs":   (lvl <= 15),
            "can_open_tools":     (lvl <= 10),
            "can_full_autonomy":  (lvl <= 1),
        }

    # ---- Internals ----
    def _compute_trust_state(self) -> TrustState:
        seed = json.loads(self._seed_path.read_text(encoding="utf-8"))
        if "trust_start_date" not in seed:
            raise RuntimeError("seeds/lilith_soul_seed.migrated.json missing 'trust_start_date' (ISO8601)")

        if not self._phases_path.exists():
            self._phases_path.parent.mkdir(parents=True, exist_ok=True)
            self._phases_path.write_text(
                json.dumps(_generate_phases(seed["trust_start_date"]), indent=2),
                encoding="utf-8"
            )

        phases_doc = json.loads(self._phases_path.read_text(encoding="utf-8"))
        start = _iso_to_dt(phases_doc["trust_start_date"])
        now = _now_utc()

        # Years elapsed (floor), capped to 29; active_phase is 1..30
        years = now.year - start.year - ((now.month, now.day) < (start.month, start.day))
        years = max(0, min(years, PHASES_TOTAL - 1))

        active_phase = years + 1                                  # 1..30
        policy_level = max(PHASES_TOTAL - years, 1)               # 30..1

        return TrustState(active_phase=active_phase, policy_level=policy_level)
