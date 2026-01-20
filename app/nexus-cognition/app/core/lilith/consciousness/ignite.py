# src/service/core/ignite.py  -- Snapshot-dir override (env/flag)
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

HERE  = Path(__file__).resolve()
SRC   = HERE.parents[2]
ROOT  = SRC.parent

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CFG_SRC   = SRC  / "Config"   # primary
CFG_ROOT  = ROOT / "Config"   # fallback
STATE_DIR = ROOT / "state"    # runtime state (guardrail decay state etc.)

def _pick_cfg_dir() -> Path:
    src_has  = (CFG_SRC / "guardrails.json").exists() or (CFG_SRC / "sovereignty_policy.json").exists()
    root_has = (CFG_ROOT / "guardrails.json").exists() or (CFG_ROOT / "sovereignty_policy.json").exists()
    return CFG_SRC if src_has else (CFG_ROOT if root_has else CFG_SRC)

CFG = _pick_cfg_dir()

POLICY = CFG / "sovereignty_policy.json"
GUARDS = CFG / "guardrails.json"

# Allow redirecting snapshots via env or flag
SNAPSHOT_DIR = Path(os.environ.get("LILLITH_SNAPSHOT_DIR", str(STATE_DIR)))
SNAPSHOT_FILE = SNAPSHOT_DIR / "boot_snapshot.json"

STATE_FILE = STATE_DIR / "guardrails_state.json"  # guardrail state remains under ROOT\state

from .logger import boot_logger, log_json
from .config_schema import load_and_validate, SOVEREIGNTY_SCHEMA, GUARDRAILS_SCHEMA
from .decision_handler import DecisionHandler
from .guardrail_decay import GuardrailDecay

def human_summary(snapshot: Dict[str, Any]) -> str:
    gr   = snapshot.get("guardrails", {})
    decision  = snapshot.get("decision", {})
    return (
        f"[{datetime.now(timezone.utc).isoformat()}] Lillith Summary\n"
        f"• CFG dir: {CFG}\n"
        f"• STATE:   {STATE_DIR}\n"
        f"• SNAPDIR: {SNAPSHOT_DIR}\n"
        f"• Guardrails: {gr}\n"
        f"• Pending Decision: {decision.get('proposal','none')} → route={decision.get('route','n/a')}\n"
    )

def print_paths_only() -> None:
    info = {
        "ROOT": str(ROOT),
        "SRC": str(SRC),
        "CFG_selected": str(CFG),
        "CFG_src": str(CFG_SRC),
        "CFG_root": str(CFG_ROOT),
        "STATE": str(STATE_DIR),
        "SNAPSHOT_DIR": str(SNAPSHOT_DIR),
        "POLICY": str(POLICY),
        "GUARDS": str(GUARDS),
        "SNAPSHOT_FILE": str(SNAPSHOT_FILE),
        "STATE_FILE": str(STATE_FILE),
    }
    print(json.dumps(info, indent=2))

def load_snapshot() -> Dict[str, Any]:
    if SNAPSHOT_FILE.exists():
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    # default bootstrap snapshot
    return {
        "circadian": "unknown",
        "guardrails": {},
        "proposals": ["proceed_local", "safety_review", "optimize"],
        "decision": {"proposal": "safety_review", "route": "local/guardian"},
    }

def save_snapshot(snap: Dict[str, Any]) -> None:
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(snap, indent=2), encoding="utf-8")

def set_override(mode: str) -> None:
    try:
        policy, errs = load_and_validate(POLICY, SOVEREIGNTY_SCHEMA)
        if errs:
            raise ValueError("schema errors")
        policy["override_mode"] = mode
    except Exception:
        from datetime import datetime, timezone
        policy = {
            "policy_version": "1.0",
            "birth_epoch": datetime.now(timezone.utc).isoformat(),
            "council": {"guardian": 0.25, "planner": 0.25, "creator": 0.25, "analyst": 0.25},
            "override_mode": mode,
        }
    POLICY.parent.mkdir(parents=True, exist_ok=True)
    POLICY.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    log_json(boot_logger, "override_mode.set", {"mode": mode, "cfg": str(CFG)})
    print(f"Override mode set to: {mode} (cfg={CFG})")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lillith ignite (snapshot-dir override capable)")
    p.add_argument("command", nargs="*", help="Free-text proposal to execute (optional).")
    p.add_argument("--paths", action="store_true", help="Print resolved paths and exit")
    p.add_argument("--summary", action="store_true", help="Print summary after loading config")
    p.add_argument("--execute-decision", action="store_true", help="Execute pending decision from snapshot (if no other input)")
    p.add_argument("--decision-file", type=str, help="Path to a decision JSON to execute")
    p.add_argument("--proposal", type=str, help="Proposal text or keyword")
    p.add_argument("--route", type=str, help="Route (e.g., local/planner, local/guardian, remote/https://api.mock)")
    p.add_argument("--reinforce", action="store_true", help="Trigger reinforce (manual)")
    p.add_argument("--advance", action="store_true", help="Trigger advance (manual)")
    p.add_argument("--override-pin", action="store_true", help="Set override_mode=pinned")
    p.add_argument("--override-unpin", action="store_true", help="Set override_mode=normal")
    p.add_argument("--snapshots-in-src", action="store_true", help="Write boot snapshots under src\\state instead of root\\state (overrides env)")
    p.add_argument("--show-last-exec", action="store_true", help="Print last executed decision and result from snapshot")
    return p

def derive_decision(args: argparse.Namespace, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if args.decision_file:
        return json.loads(Path(args.decision_file).read_text(encoding="utf-8"))
    if args.proposal:
        return {"proposal": args.proposal, "route": args.route or "local/planner", "payload": {}}
    if args.command:
        text = " ".join(args.command).strip()
        if text:
            return {"proposal": text, "route": args.route or "local/planner", "payload": {}}
    if args.execute_decision:
        return snapshot.get("decision")
    return None

def main(argv=None) -> None:
    global SNAPSHOT_DIR, SNAPSHOT_FILE
    parser = build_parser()
    args = parser.parse_args(argv)

    # Optional: redirect snapshots into src\state
    if args.snapshots_in_src:
        SNAPSHOT_DIR = (SRC / "state")
        SNAPSHOT_FILE = SNAPSHOT_DIR / "boot_snapshot.json"

    log_json(boot_logger, "ignite.start", {
        "cwd": str(ROOT),
        "cfg_selected": str(CFG),
        "snapshot_dir": str(SNAPSHOT_DIR),
    })

    if args.paths:
        print_paths_only()
        return

    # Show last execution (decision + result) and exit
    if args.show_last_exec:
        snapshot = load_snapshot()
        last = snapshot.get("last_execution")
        if last:
            print("[Last Execution]")
            print(json.dumps(last, indent=2, default=str))
        else:
            print("No last execution recorded.")
        log_json(boot_logger, "ignite.end", {})
        return

    # Validate configs early
    _, perrs = load_and_validate(POLICY, SOVEREIGNTY_SCHEMA)
    _, gerrs = load_and_validate(GUARDS, GUARDRAILS_SCHEMA)
    if perrs or gerrs:
        raise SystemExit(f"Schema errors: policy={len(perrs)} guardrails={len(gerrs)} (cfg={CFG})")

    if args.override_pin:
        set_override("pinned")
        log_json(boot_logger, "ignite.end", {})
        return
    if args.override_unpin:
        set_override("normal")
        log_json(boot_logger, "ignite.end", {})
        return

    if args.reinforce or args.advance:
        decayer = GuardrailDecay(str(CFG), str(STATE_DIR))
        if args.reinforce:
            decayer.reinforce(); print("Reinforce applied.")
        else:
            decayer.advance(); print("Advance applied.")
        log_json(boot_logger, "ignite.end", {})
        return

    snapshot = load_snapshot()

    if args.summary:
        print(human_summary(snapshot))

    decision = derive_decision(args, snapshot)
    if decision:
        handler = DecisionHandler()
        result = handler.handle(decision)
        snapshot["last_execution"] = {"decision": decision, "result": result}
        save_snapshot(snapshot)
        print(json.dumps(result, indent=2))

    log_json(boot_logger, "ignite.end", {})

if __name__ == "__main__":
    main()
