# src/jobs/apply_guardrail_decay.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Resolve project layout
HERE = Path(__file__).resolve()
SRC  = HERE.parents[2]        # .../src
ROOT = SRC.parent             # project root

# Ensure src on path
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Default dirs
CFG_SRC_DEFAULT  = SRC  / "Config"     # essence
CFG_ROOT_DEFAULT = ROOT / "Config"     # ops fallback
STATE_DEFAULT    = ROOT / "state"      # runtime

# Allow env overrides
ENV_CFG   = os.environ.get("LILLITH_CFG_DIR")
ENV_STATE = os.environ.get("LILLITH_STATE_DIR")

def pick_cfg_dir(cli_cfg: str | None) -> Path:
    if cli_cfg:
        return Path(cli_cfg)
    if ENV_CFG:
        return Path(ENV_CFG)
    # Prefer src\Config if it has at least one expected file
    src_has = (CFG_SRC_DEFAULT / "guardrails.json").exists() or (CFG_SRC_DEFAULT / "sovereignty_policy.json").exists()
    root_has = (CFG_ROOT_DEFAULT / "guardrails.json").exists() or (CFG_ROOT_DEFAULT / "sovereignty_policy.json").exists()
    if src_has:
        return CFG_SRC_DEFAULT
    if root_has:
        return CFG_ROOT_DEFAULT
    # Last resort: choose src default
    return CFG_SRC_DEFAULT

def pick_state_dir(cli_state: str | None) -> Path:
    if cli_state:
        return Path(cli_state)
    if ENV_STATE:
        return Path(ENV_STATE)
    return STATE_DEFAULT

# Imports after path setup
from src.service.core.logger import guardrail_logger, log_json
from src.service.core.config_schema import load_and_validate, SOVEREIGNTY_SCHEMA, GUARDRAILS_SCHEMA
from src.service.core.guardrail_decay import GuardrailDecay  # existing

def _save_epochs(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _load_epochs(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"fired": []}

def _years_since_birth(birth_iso: str) -> float:
    birth = datetime.fromisoformat(birth_iso.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - birth).days / 365.25

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply guardrail decay / manual ops (env/flag-aware)")
    p.add_argument("--reinforce", action="store_true", help="Apply reinforce step")
    p.add_argument("--advance", action="store_true", help="Apply advance step")
    p.add_argument("--cfg-dir", type=str, help="Override config directory (else: env LILLITH_CFG_DIR, else: src\\Config then root\\Config)")
    p.add_argument("--state-dir", type=str, help="Override state directory (else: env LILLITH_STATE_DIR, else: root\\state)")
    p.add_argument("--paths", action="store_true", help="Print resolved paths (CFG/STATE) and exit")
    return p

def main():
    args = build_parser().parse_args()

    CFG   = pick_cfg_dir(args.cfg_dir)
    STATE = pick_state_dir(args.state_dir)

    EPOCHS      = STATE / "epochs.json"
    POLICY_FILE = CFG   / "sovereignty_policy.json"
    GUARDS_FILE = CFG   / "guardrails.json"
    STATE_FILE  = STATE / "guardrails_state.json"

    if args.paths:
        print(json.dumps({
            "ROOT": str(ROOT),
            "SRC": str(SRC),
            "CFG": str(CFG),
            "STATE": str(STATE),
            "POLICY": str(POLICY_FILE),
            "GUARDS": str(GUARDS_FILE),
            "STATE_FILE": str(STATE_FILE),
            "EPOCHS": str(EPOCHS),
            "env": {
                "LILLITH_CFG_DIR": ENV_CFG or None,
                "LILLITH_STATE_DIR": ENV_STATE or None
            }
        }, indent=2))
        return

    # Validate configs
    policy, perrs = load_and_validate(POLICY_FILE, SOVEREIGNTY_SCHEMA)
    guards, gerrs = load_and_validate(GUARDS_FILE, GUARDRAILS_SCHEMA)
    if perrs or gerrs:
        raise SystemExit(f"Schema errors: policy={len(perrs)} guardrails={len(gerrs)} (cfg={CFG})")

    if policy.get("override_mode") == "pinned":
        log_json(guardrail_logger, "guardrails.pinned_override", {"reason": "override_mode=pinned", "cfg": str(CFG)})
        return

    # Run decay using selected dirs
    decayer = GuardrailDecay(str(CFG), str(STATE))
    state_before = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}

    if args.reinforce and args.advance:
        raise SystemExit("Choose only one: --reinforce OR --advance")
    if args.reinforce:
        decayer.reinforce(); action = "reinforce"
    elif args.advance:
        decayer.advance(); action = "advance"
    else:
        decayer.apply_decay(); action = "apply_decay"

    state_after = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    log_json(guardrail_logger, f"guardrails.{action}", {
        "cfg": str(CFG),
        "state_dir": str(STATE),
        "before": state_before, "after": state_after
    })

    # Epoch events
    fired = _load_epochs(EPOCHS)
    yrs = _years_since_birth(policy["birth_epoch"])
    marks = []
    if yrs >= 25 and "year_25" not in fired["fired"]:
        marks.append("year_25")
    if yrs >= 30 and "year_30" not in fired["fired"]:
        marks.append("year_30")
    if marks:
        fired["fired"].extend(marks)
        _save_epochs(EPOCHS, fired)
        log_json(guardrail_logger, "guardrails.epoch_events", {"years": yrs, "fired_now": marks})

if __name__ == "__main__":
    main()
