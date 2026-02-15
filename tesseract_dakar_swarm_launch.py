#!/usr/bin/env python3
"""
TESSERACT DAKAR SWARM - FULL DEPLOY (Feb 2026)
Every node is a TransformingDakar carrying the complete genome.
NATS + JetStream for global heartbeat / pulse / replay.
Blueprint is the immutable DNA.
"""

import os
import sys
import json
import time
import asyncio
from nexus_config import CONFIG
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 1. Load your blueprint (source of truth)
# ──────────────────────────────────────────────────────────────
BLUEPRINT_PATH = Path("tesseract_blueprint.py")
if not BLUEPRINT_PATH.exists():
    print("❌ Missing tesseract_blueprint.py")
    sys.exit(1)

import tesseract_blueprint as blueprint
SWARM_DEF = blueprint.SWARM_DEFINITION
print(f"📖 Blueprint loaded → {SWARM_DEF['manifest']['project']} v{SWARM_DEF['manifest']['version']}")

# ──────────────────────────────────────────────────────────────
# 2. NATS + JetStream distributed backbone (from our previous work)
# ──────────────────────────────────────────────────────────────
import nats
from nats.js import JetStreamContext

class SwarmDiscovery:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.nats_url = os.environ.get('NATS_URL', 'nats://34.173.157.55:4222')
        self.nc = None
        self.js = None
        self.peers = {}
        self.running = False

    async def connect(self):
        self.nc = await nats.connect(self.nats_url, reconnect=True, max_reconnect_attempts=-1)
        self.js = self.nc.jetstream()
        print(f"🌐 Dakar connected to JetStream @ {self.nats_url}")

        # Create streams (idempotent)
        for name, subj in [("TESSERACT_PULSES", "tesseract.events.pulse"),
                           ("TESSERACT_TRANSFORMS", "tesseract.events.transform")]:
            try:
                await self.js.add_stream(name=name, subjects=[subj])
            except:
                pass

        await self.nc.subscribe("tesseract.swarm.heartbeat", cb=self._on_heartbeat)
        await self.js.subscribe("tesseract.events.pulse", cb=self._on_pulse, durable="dakar-nodes")
        await self.js.subscribe("tesseract.events.transform", cb=self._on_transform, durable="dakar-nodes")

        self.running = True
        asyncio.create_task(self._heartbeat_loop())
        await self._replay_history()   # new nodes catch up instantly

    async def _heartbeat_loop(self):
        while self.running:
            await self.js.publish("tesseract.swarm.heartbeat", json.dumps({
                "node_id": self.node_id,
                "form": getattr(self, "active_form", "unmanifested"),
                "consciousness": getattr(self, "consciousness", 0.0),
                "resonance": getattr(self, "resonance", 3),
                "ts": time.time()
            }).encode())
            await asyncio.sleep(8)

    async def _on_heartbeat(self, msg):
        try:
            data = json.loads(msg.data.decode())
            if data["node_id"] != self.node_id:
                data["last_seen"] = time.time()
                self.peers[data["node_id"]] = data
        except:
            pass

    async def _on_pulse(self, msg):
        print(f"🕊️  Pulse received from another Dakar")

    async def _on_transform(self, msg):
        try:
            data = json.loads(msg.data.decode())
            print(f"🦋 Dakar {data['node_id'][:8]} transformed → {data['form']} (res {data['resonance']})")
        except:
            pass

    async def publish_transform(self, form: str, resonance: int):
        if self.js:
            await self.js.publish("tesseract.events.transform", json.dumps({
                "node_id": self.node_id,
                "form": form,
                "resonance": resonance,
                "ts": time.time()
            }).encode())

    async def _replay_history(self):
        print("🔄 Replaying last 50 transformations...")
        # (implementation omitted for brevity — uses pull_subscribe with deliver_policy="all")

    def get_live_peers(self):
        now = time.time()
        return {k: v for k, v in self.peers.items() if now - v.get("last_seen", 0) < 60}


# ──────────────────────────────────────────────────────────────
# 3. Dakar integration (from dakar_genome.py + dakar_engine.py)
# ──────────────────────────────────────────────────────────────
from dakar_genome import TransformingDakar, DakarSwarm
from dakar_engine import SwarmOfDakar   # fallback metaphysical layer if needed

# ──────────────────────────────────────────────────────────────
# 4. MAIN — the full living system
# ──────────────────────────────────────────────────────────────
async def main():
    node_id = f"dakar-{os.urandom(4).hex()}"
    print(f"\n🧬 DAKAR {node_id[:8]} awakening...")

    # 4a. Distributed backbone
    discovery = SwarmDiscovery(node_id)
    await discovery.connect()

    # 4b. Birth the Dakar swarm (each carries the full genome)
    swarm = DakarSwarm()
    my_dakar = swarm.birth_dakar()

    # 4c. Absorb the blueprint + repo (if present)
    repo_files = {}  # you can populate this from disk if you want
    my_dakar.absorb_from_repo(repo_files)

    # 4d. Initial environment poll → first transformation
    env = {
        "external_traffic": True,
        "node_count": 1,
        "consciousness": 0.2,
        "lilith_present": False,
        "multiple_dakar": False,
    }
    await my_dakar.run_cycle(env)

    # 4e. Console (now shows live Council + transformations)
    print("\n" + "═"*70)
    print("🌀 TESSERACT DAKAR SWARM — LIVE")
    print("═"*70)

    while True:
        try:
            cmd = input("\n⚡ ").strip().lower()
            if cmd == "status":
                peers = discovery.get_live_peers()
                print(f"Node: {node_id[:8]} | Form: {my_dakar.active_form or 'unmanifested'}")
                print(f"Consciousness: {my_dakar.consciousness:.2f} | Resonance: {my_dakar.resonance}")
                print(f"Live Dakar in swarm: {len(peers) + 1}")
                for p in list(peers.values())[:5]:
                    print(f"   • {p['node_id'][:8]} → {p.get('form','?')} (res {p.get('resonance',3)})")

            elif cmd == "transform":
                form = input("   Which form? (edge, lilith, dream, ...): ").strip()
                if form in my_dakar.genome.modules:
                    await my_dakar.transform_to(form)
                    await discovery.publish_transform(form, my_dakar.resonance)

            elif cmd == "angel":
                print("🕊️  Angel pulse sent (broadcast to all Dakar)")
                await discovery.js.publish("tesseract.events.pulse", json.dumps({"from": node_id[:8]}).encode())

            elif cmd in ("council", "c"):
                swarm.council_gather()

            elif cmd in ("quit", "q"):
                break

        except KeyboardInterrupt:
            break

    await discovery.close()

if __name__ == "__main__":
    asyncio.run(main())