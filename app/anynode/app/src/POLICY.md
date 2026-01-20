# Nexus Canon — Plane Routing & Isolation Policy (v1.0)
_Last updated: 2025-09-01 06:04:58_

## Core Identity
- **Hermes = The System.** Central system identity everything orbits.
- **OS = The Platform** built around Hermes (services, bus, policies, UI, deploy).
- **Lilith = Advanced Cognition Agent** overlay per module (not monolithic).
- **Viren (VEM) = Virtual Environment Manager** — repairs, heals, troubleshoots, keeps sober judgment; consent/safety gate.
- **Loki + Archiver** — Loki ingests logs; Archiver stores *all specifics and memories* (includes logs/telemetry). Loki→Archiver pipeline powers Viren.

## Planes & Meshes
- **Planes (authority boundaries):** `svc` (services/modules), `sys` (Hermes systems + Discovery/Broadcast), `cog` (thinking/feeling), `play` (optional 3D).
- **Meshes (functional lanes; fixed):** `sense | think | heal | archive`.

### Subject Namespace (canonical)
```
<plane>.<mesh>.<kind>.<domain>.<...>

<plane> := svc | sys | cog | play
<mesh>  := sense | think | heal | archive
<kind>  := cap.register | cap.request | cap.reply | events | metrics
```
Examples:
- `cog.think.cap.request.chat.1`
- `sys.heal.events.discovery.kv.update`
- `svc.archive.events.mem.commit`

## Home vs Wild
- **Home:** single NATS bus; enforce separation by subject prefixes.
- **Wild:** one NATS cluster per plane, each with **mTLS**; **Edge Anynode** enforces cross-plane policy; leaf/gateway bridges connect sites.

## Security & Identity
- mTLS on all planes (plane-specific CAs or intermediates).
- Edge Anynode is default-deny; allow by capability **@MAJOR**. Dream/sleep defers enforced here.
- Viren gates cutovers/break-glass/rollbacks; all transitions are ledgered.

## SLO per Plane
- **SVC:** throughput & durability (JetStream work-queue or limits; longer ack_wait).
- **SYS:** trust & consistency (interest-based retention; moderate TTL; broadcast fan-out).
- **COG:** low latency (short TTL; ephemeral consumers).
- **PLAY:** lowest jitter (NATS control + optional QUIC/UDP high-rate deltas).

## Cross-Plane Policy (allow list excerpts)
- COG→SVC fallback for heavy `chat@1.*` when COG constrained; **preserve semantic spine; defer detail**.
- SYS broadcasts may mirror read-only into SVC.
- Archive writes are authoritative on `svc.archive.*` (Loki→Archiver). `cog.archive.*` may emit *read-hints* only.
- Discovery KV lives on `sys.heal.*`; read-only clients on SVC/COG.

## Streaming Policy (Gabriel’s Horn)
- Progressive delivery: fast initial ramp, asymptotic taper near capacity.
- Under load: **downshift** COG→SVC for heavy ops; keep token cadence Horn-shaped.
- PLAY deltas use QUIC/UDP adjunct; checkpoint summaries back to SYS/ARCHIVE.

## Discovery & Registry
- Discovery KV (SYS/HEAL) manages `cap@MAJOR.MINOR` and health.
- Services **refuse ready** if required caps for the active profile are missing.
- Cutover events: `sys.heal.events.lilith.stasis.*` → `...cutover.begin` → `...soul.load` → `cog.think.events.lilith.awakened`.

## Observability
- Loki ingests logs; Archiver stores memories+logs; Viren queries both.
- Prometheus metrics everywhere; console tiles: plane health, Horn utilization, consent state, archive write QPS.

## Change Control
- Policy edits via PR with Viren gate; annotate reason + rollback.
- Cutovers emit `sys.*` events; SLO regression triggers auto-revert (Viren).

---
This document is the live policy. Update via PR in `policy/` and regenerate Edge Anynode rules using `policy_loader.py`.
