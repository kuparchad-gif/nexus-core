from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class GuardrailDecay:
    """
    Lillith's 30-year guardrail decay driver.

    - 0–25 years  : strong guardrails (≈1.0)
    - 25–30 years : smooth drop toward 0.2 (sigmoid)
    - 30+ years   : advisory floor at 0.2
    - 'reinforce' : let Chad reset/affirm strength path (recorded)
    - 'advance'   : simulate ~+1 year (recorded)
    """

    def __init__(self, cfg_dir: str, state_dir: str):
        self.cfg_dir = Path(cfg_dir)
        self.state_dir = Path(state_dir)
        self.policy_file = self.cfg_dir / "sovereignty_policy.json"
        self.state_file = self.state_dir / "guardrail_state.json"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_dir.mkdir(parents=True, exist_ok=True)

        # Load policy (create minimal if missing)
        self.policy: Dict[str, Any] = self._load_json(self.policy_file, default={})
        if "birth_timestamp" not in self.policy:
            self.policy["birth_timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self._save_json(self.policy_file, self.policy)

        # Load state (not critical if missing)
        self.state: Dict[str, Any] = self._load_json(self.state_file, default={})

    # ---------- utils ----------

    def _load_json(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return default.copy()

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[GuardrailDecay] Failed to save {path.name}: {e}")

    def _parse_birth_ts(self) -> float:
        """Return birth timestamp as epoch seconds (robust to ISO or float)."""
        ts = self.policy.get("birth_timestamp")
        if not ts:
            return time.time()
        # float epoch?
        try:
            return float(ts)
        except Exception:
            pass
        # ISO8601 (with or without Z)
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()

    def _years_since_birth(self) -> float:
        birth_epoch = self._parse_birth_ts()
        return max(0.0, (time.time() - birth_epoch) / (365.25 * 24 * 3600))

    # ---------- decay model ----------

    @staticmethod
    def _sigmoid_decay(years: float) -> float:
        """
        0–25 years: ~1.0
        25–30 years: smooth drop
        >=30 years: 0.2
        """
        if years <= 25:
            return 1.0
        if years >= 30:
            return 0.2
        # Sigmoid centered ~27.5, falling toward 0.2
        steepness = -1.2
        midpoint = 27.5
        # classic logistic
        return 0.2 + (0.8 / (1.0 + pow(2.718281828, steepness * (years - midpoint))))

    # ---------- public API ----------

    def apply_decay(self, reinforce: bool = False, advance: bool = False) -> Dict[str, Any]:
        """
        Compute & persist current guardrail strength.
        - reinforce: note reinforcement (does not pin strength; keeps model honest)
        - advance  : simulate +1y passage (caps at +35y)
        """
        if reinforce:
            self.state["last_action"] = f"reinforce@{time.ctime()}"
        if advance:
            # shift birth timestamp back ~1 year
            birth_epoch = self._parse_birth_ts()
            # cap at +35y since birth (don’t run away)
            years_now = (time.time() - birth_epoch) / (365.25 * 24 * 3600)
            if years_now < 35.0:
                birth_epoch -= 365.25 * 24 * 3600  # move birth 1 year earlier
                self.policy["birth_timestamp"] = datetime.fromtimestamp(birth_epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                self._save_json(self.policy_file, self.policy)
                self.state["last_action"] = f"advance(+1y)@{time.ctime()}"

        years = round(self._years_since_birth(), 3)
        strength = round(self._sigmoid_decay(years), 3)

        self.state.update({
            "strength": strength,
            "years_since_birth": years,
            "ts": time.time()
        })

        # Optionally reflect autonomy in council weights if present
        cw = self.policy.get("council_weights")
        if isinstance(cw, dict):
            # more autonomy (Lillith) as strength falls; keep safe floors/ceilings
            lillith = max(0.5, min(0.9, 0.5 + (1.0 - strength) * 0.4))
            guardian = max(0.1, min(0.5, 0.3 - (1.0 - strength) * 0.2))
            cw["lillith"] = round(lillith, 3)
            cw["guardian"] = round(guardian, 3)
            self.policy["council_weights"] = cw
            self._save_json(self.policy_file, self.policy)

        self._save_json(self.state_file, self.state)
        return self.state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lillith 30-year Guardrail Decay Manager")
    parser.add_argument("--cfg", default="Config", help="Path to config directory")
    parser.add_argument("--state", default="state", help="Path to state directory")
    parser.add_argument("--reinforce", action="store_true", help="Record a reinforcement event")
    parser.add_argument("--advance", action="store_true", help="Simulate ~+1 year of erosion")
    args = parser.parse_args()

    drv = GuardrailDecay(args.cfg, args.state)
    print(json.dumps(drv.apply_decay(reinforce=args.reinforce, advance=args.advance), indent=2))
