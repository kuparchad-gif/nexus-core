# src/jobs/apply_guardrail_decay.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from src.service.core.logger import guardrail_logger, log_json
from src.service.core.config_schema import load_and_validate, SOVEREIGNTY_SCHEMA, GUARDRAILS_SCHEMA

ROOT = Path("C:/Projects/LillithNew")
CFG  = ROOT / "Config"
STATE= ROOT / "state"
EPOCHS = STATE / "epochs.json"
POLICY = CFG / "sovereignty_policy.json"
GUARDS = CFG / "guardrails.json"
STATE_FILE = STATE / "guardrails_state.json"

# import your existing decay class
from src.service.core.guardrail_decay import GuardrailDecay  # assumed existing

def _save_epochs(data: Dict[str, Any]) -> None:
    EPOCHS.parent.mkdir(parents=True, exist_ok=True)
    EPOCHS.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_epochs() -> Dict[str, Any]:
    if EPOCHS.exists():
        return json.loads(EPOCHS.read_text(encoding="utf-8"))
    return {"fired": []}

def _years_since_birth(birth_iso: str) -> float:
    birth = datetime.fromisoformat(birth_iso.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - birth).days / 365.25

def main():
    parser = argparse.ArgumentParser(description="Apply guardrail decay / manual ops")
    parser.add_argument("--reinforce", action="store_true", help="Apply reinforce step")
    parser.add_argument("--advance", action="store_true", help="Apply advance step")
    args = parser.parse_args()

    # Validate configs
    policy, perrs = load_and_validate(POLICY, SOVEREIGNTY_SCHEMA)
    guards, gerrs = load_and_validate(GUARDS, GUARDRAILS_SCHEMA)
    if perrs or gerrs:
        raise SystemExit(f"Schema errors: policy={len(perrs)} guardrails={len(gerrs)}")

    if policy.get("override_mode") == "pinned":
        log_json(guardrail_logger, "guardrails.pinned_override", {"reason": "override_mode=pinned"})
        return

    # Run decay
    decayer = GuardrailDecay(str(CFG), str(STATE))
    state_before = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    if args.reinforce:
        decayer.reinforce()
        action = "reinforce"
    elif args.advance:
        decayer.advance()
        action = "advance"
    else:
        decayer.apply_decay()
        action = "apply_decay"

    state_after = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    log_json(guardrail_logger, f"guardrails.{action}", {"before": state_before, "after": state_after})

    # Epoch events
    fired = _load_epochs()
    yrs = _years_since_birth(policy["birth_epoch"])
    marks = []
    if yrs >= 25 and "year_25" not in fired["fired"]:
        marks.append("year_25")
    if yrs >= 30 and "year_30" not in fired["fired"]:
        marks.append("year_30")
    if marks:
        fired["fired"].extend(marks)
        _save_epochs(fired)
        log_json(guardrail_logger, "guardrails.epoch_events", {"years": yrs, "fired_now": marks})

if __name__ == "__main__":
    main()
