from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
from typing import Dict, Any

from .firewall import SmartFirewall
from .tripwire import Tripwire
from .db_llm_gate import DBLLMGate
from .self_destruct import SelfDestruct

class AnyNode:
    """Edge node with firewall + tripwire + DB-backed LLM gate + self-destruct."""
    def __init__(self, root: Path | None = None):
        self.root = root or Path(os.environ.get("EDGE_ROOT", Path.cwd()))
        self.firewall = SmartFirewall()
        self.tripwire = Tripwire(self.root)
        self.gate = DBLLMGate(self.root / "edge.db")
        self.destruct = SelfDestruct(self.root)

    def status(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "fw_rules": self.firewall.count(),
            "tripwire_integrity": self.tripwire.status(),
            "db_exists": (self.root / "edge.db").exists(),
        }

    def handle(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Firewall first
        verdict = self.firewall.inspect(packet)
        if verdict["block"]:
            return {"ok": False, "reason": "firewall_block", "verdict": verdict}
        # 2) Tripwire periodic check
        tw = self.tripwire.maybe_check()
        if tw.get("compromised"):
            # self-destruct if configured
            if os.environ.get("EDGE_SELF_DESTRUCT", "1") == "1":
                self.destruct.execute(reason="tripwire_compromised")
            return {"ok": False, "reason": "compromised", "tripwire": tw}
        # 3) DB LLM gate for query-type packets
        if packet.get("type") == "query":
            ans = self.gate.answer(packet.get("q",""), meta=packet.get("meta", {}))
            return {"ok": True, "answer": ans, "tw": tw}
        return {"ok": True, "echo": packet, "tw": tw}

def main():
    p = argparse.ArgumentParser(description="Edge AnyNode")
    p.add_argument("--status", action="store_true")
    p.add_argument("--packet", type=str, help="JSON packet to handle")
    p.add_argument("--root", type=str, help="Edge root override")
    args = p.parse_args()

    node = AnyNode(Path(args.root) if args.root else None)
    if args.status:
        print(json.dumps(node.status(), indent=2)); return
    if args.packet:
        pkt = json.loads(args.packet)
        print(json.dumps(node.handle(pkt), indent=2)); return
    print(json.dumps({"usage": "use --status or --packet '{...}'"}, indent=2))

if __name__ == "__main__":
    main()
