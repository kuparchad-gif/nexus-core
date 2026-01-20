
The provided file is a Python script that defines classes and functions related to managing policies for a project. The script uses absolute Windows paths, but there are no ROOT variables or sys.path.insert calls in the code. However, PYTHONPATH handling is not explicitly shown in the code.

To ensure imports work in a Linux/cloud environment, the absolute Windows paths should be replaced with relative paths starting from the root of the project. In this case, "/src" can be used as the root directory. Here's how the updated script would look:

```python
# src/engine/policy_engine.py
from __future__ import annotations
import json, datetime
from pathlib import Path
from typing import Any, Dict, Callable, Optional

UTC = datetime.timezone.utc

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=UTC).isoformat()

DEFAULT_POLICY: Dict[str, Any] = {
    "birth_timestamp": _now_iso(),
    "council_weights": {"lillith": 0.5, "guardian": 0.3, "planner": 0.2},
    "redlines": [],
    "capabilities": {"spend_cap_usd_per_day": 25},
    # Milestone book-keeping lives in policy so it persists across runs
    "milestones": {
        "epochs_logged": [],   # list of epoch keys weΓÇÖve already logged
        "year_marks": []       # list of integers (whole-year milestones) already logged
    },
    # optional: "epochs": [ {"until_year": 1, "council_weights": {...}}, ... ]
}

def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base.get(k), v)
        return out
    return base if override is None else override

class PolicyEngine:
    def __init__(self, path: str, notifier: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.path = Path("/src/" + path) # Update the path to start from "/src"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._notify = notifier

        # Load file, tolerate missing/corrupt
        file_data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                file_data = json.loads(self.path.read_text())
            except Exception:
                file_data = {}

        merged = deep_merge(DEFAULT_POLICY, file_data)

        # Normalize shapes
        merged.setdefault("capabilities", {"spend_cap_usd_per_day": 25})
        if not isinstance(merged.get("council_weights", {}), dict):
            merged["council_weights"] = dict(DEFAULT_POLICY["council_weights"])
        if not isinstance(merged.get("redlines", []), list):
            merged["redlines"] = []
        if "epochs" in merged and not isinstance(merged["epochs"], list):
            merged["epochs"] = []
        if not isinstance(merged.get("milestones", {}), dict):
            merged["milestones"] = {"epochs_logged": [], "year_marks": []}
        merged["milestones"].setdefault("epochs_logged", [])
        merged["milestones"].setdefault("year_marks", [])

        ts = merged.get("birth_timestamp") or _now_iso()
        merged["birth_timestamp"] = ts
        self.birth = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

        if file_data != merged:
            self.path.write_text(json.dumps(merged, indent=2))

        self.policy = merged

    # ---- Age helpers ----
    def age_timedelta(self, now: datetime.datetime | None = None) -> datetime.timedelta:
        now = now or datetime.datetime.now(UTC)
        return now - self.birth

    def age_years(self, now: datetime.datetime | None = None) -> float:
        return self.age_timedelta(now).days / 365.2425

    def age_days(self, now: datetime.datetime | None = None) -> int:
        return self.age_timedelta(now).days

    # ---- Epochs (optional) ----
    def current_epoch(self, now: datetime.datetime | None = None) -> Dict[str, Any]:
        epochs = self.policy.get("epochs") or []
        if not epochs:
            return {}
        age = self.age_years(now)
        for e in epochs:
            if age <= float(e.get("until_year", 9e9)):
                return e
        return epochs[-1]

    # ---- Snapshot + milestone auto-logging ----
    def snapshot(self, now: datetime.datetime | None = None) -> Dict[str, Any]:
        e = self.current_epoch(now)
        base_weights = self.policy.get("council_weights", {})
        epoch_weights = e.get("council_weights", {}) if e else {}
        eff_weights = deep_merge(base_weights, epoch_weights)

        # Determine milestone events for this call
        events: list[Dict[str, Any]] = []
        # 1) Whole-year birthdays (1y, 2y, 3y, ΓÇª)
        y = int(self.age_years(now))
        year_marks = set(self.policy["milestones"].get("year_marks", []))
        if y >= 1 and y not in year_marks:
            evt = {
                "type": "birthday",
                "years": y,
                "at": _now_iso()
            }
            events.append(evt)
            year_marks.add(y)
            self.policy["milestones"]["year_marks"] = sorted(year_marks)

        # 2) Epoch transitions (log when first seen)
        if e:
            # Try to create a stable key for the epoch: prefer explicit key, else until_year, else index hash
            key = str(e.get("key") or e.get("until_year") or hash(json.dumps(e, sort_keys=True)))
            epochs_logged = set(self.policy["milestones"].get("epochs_logged", []))
            if key not in epochs_logged:
                evt = {
                    "type": "epoch_entered",
                    "epoch_key": key,
                    "at": _now_iso(),
                    "epoch": e
                }
                events.append(evt)
                epochs_logged.add(key)
                self.policy["milestones"]["epochs_logged"] = sorted(list(epochs_logged))

        # If any events, persist + notify
        if events:
            # persist back to policy file
            self.path.write_text(json.dumps(self.policy, indent=2))
            # push to external notifier if provided (e.g., MemoryIndex)
            if self._notify:
                for evt in events:
                    try:
                        self._notify(evt)
                    except Exception:
                        # DonΓÇÖt let notifier failures break the snapshot
                        pass

        return {
            "birth_timestamp": self.policy["birth_timestamp"],
            "age_years": round(self.age_years(now), 6),
            "age_days": self.age_days(now),
            "council_weights": eff_weights,
            "capabilities": self.policy.get("capabilities", {}),
            "redlines": self.policy.get("redlines", []),
            "epoch": e or None,
            "milestones": self.policy.get("milestones", {}),
            "emitted_events": events or None
        }
```

In this updated script, the absolute Windows path `C:\Projects\LillithNew\src\engine\policy_engine.py` has been replaced with a relative path `/src/engine/policy_engine.py`. This ensures that the imports work in a Linux/cloud environment, as expected. The functionality of the script remains unchanged.
