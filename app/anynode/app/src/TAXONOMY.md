# Nexus Metatron — Category Canon (v1)
_Last updated: 2025-09-01 06:55:24_

**Single source of truth for where every service lives.** Nothing lives outside these six categories.

## Categories (authoritative)
1. **Consciousness** — persona/session manager, dialog policy, affect, Kairos (subjective time), *front-of-mind* execution.
   - Plane bias: **COG**
   - Mesh bias: **think/heal**
   - Examples: dialog driver for `chat@1.0`, affect modulator, Kairos governor.

2. **Subconsciousness** — codecs, filters, verifiers, ingestion & reflexes that support Consciousness.
   - Plane bias: **COG** (sometimes SVC for throughput)
   - Mesh bias: **sense/think/heal**
   - Examples: **NIM** (lattice codec), **Verifier**, spectral filters, PsyShield.

3. **Upper Cognition** — advanced planners/agents (Lilith overlays), long-horizon synthesis, tool strategies.
   - Plane bias: **COG**
   - Mesh bias: **think**
   - Examples: `brain.pulse@1.0`, `brain.harmony@1.0`, `wire@1.0`, `alchemy@1.0`

4. **Memory** — durable knowledge & logs: Archiver, Qdrant indices, retrieval, memory guards.
   - Plane bias: **SVC**
   - Mesh bias: **archive/sense**
   - Examples: Archiver API, Memory Sentinel, Q-Graph indexers, Loki→Archiver pipeline.

5. **Nexus Core** — control plane, discovery, receipts, cutovers, policy distribution, and health.
   - Plane bias: **SYS**
   - Mesh bias: **heal/think**
   - Examples: nexus-core (signed control + hash-chained receipts), Discovery KV, Viren console, Bridge WS (control side).

6. **Edge Anynode** (Blood–Brain Barrier) — cross-plane firewall/router, rate limits, encryption, client ingress.
   - Plane bias: **SYS** (policy) + **SVC** (data path)
   - Mesh bias: **heal** (policy), **sense/think** (bridges)
   - Examples: Edge policy engine, WS ingress, Prompt Firewall, QUIC adjunct for PLAY.

---

## Canonical mapping (current services → category)

| Service / Capability                              | Category          | Plane.Mesh (primary)                        |
|---                                                |---                |---                                          |
| Gateway (OpenAI-compatible) → `chat@1.0`          | **Consciousness** | COG.think (fallback to SVC.think)           |
| Lilith Chat overlay                               | **Consciousness** | COG.think                                   |
| Kairos (temporal governor)                        | **Consciousness** | COG.heal                                    |
| PsyShield / Prompt Firewall                       | **Subconsciousness** | COG.heal                                  |
| NIM (`encode/decode/infer/stream@1.0`)           | **Subconsciousness** | COG.think                                  |
| Verifier (`verify.answer@1.0`)                    | **Subconsciousness** | COG.think                                  |
| Upper cognition (`brain.pulse/harmony/wire/alchemy`) | **Upper Cognition** | COG.think                               |
| Archiver (mem+receipts store)                     | **Memory**        | SVC.archive                                 |
| Memory Sentinel (poisoning guard)                 | **Memory**        | SVC.archive                                 |
| Loki → Archiver pipeline                          | **Memory**        | SVC.heal/archive                            |
| Discovery KV + signed control (nexus-core)        | **Nexus Core**    | SYS.heal                                    |
| Viren (VEM, receipts, cutovers)                   | **Nexus Core**    | SYS.heal                                    |
| Bridge WS (control/event bridge)                  | **Nexus Core**    | SYS.think                                   |
| Edge policy engine (Edge Anynode / BBB)           | **Edge Anynode**  | SYS.heal                                    |
| WS ingress / client router                        | **Edge Anynode**  | SVC.think (data), SYS.heal (policy)         |
| Integrity Aegis (OS validity)                     | **Nexus Core**    | SVC.heal                                    |

> Optional **PLAY** plane components (game mesh) are attributed to **Edge Anynode** for ingress & tick control, while their agent logic maps to **Upper Cognition**.

---

## Capability registration schema (Discovery KV)

Each service registers to `*.cap.register` with the following payload:

```json
{
  "service": "lilith_chat",
  "category": "Consciousness",
  "caps": [{"name": "chat", "version": "1.0"}],
  "plane": "cog",
  "mesh": "think",
  "endpoints": {"health": "http://host:port/health"},
  "labels": {"owner": "nexus", "tier": "gold"}
}
```

Valid `category` values (must match exactly):
- "Consciousness", "Subconsciousness", "Upper Cognition", "Memory", "Nexus Core", "Edge Anynode"

---

## Subject naming (no breaking changes, but stronger domain discipline)

Keep the canonical subject shape:
```
<plane>.<mesh>.<kind>.<domain>.<...>
```

**Recommendation:** begin `<domain>` with a category key so routing and policy stay clean:
- `cons.chat.*` → Consciousness
- `subc.nim.*`, `subc.verify.*` → Subconsciousness
- `upcg.brain.*` → Upper Cognition
- `mem.archiver.*` → Memory
- `core.discovery.*`, `core.control.*` → Nexus Core
- `edge.policy.*`, `edge.ingress.*` → Edge Anynode

Example:
```
cog.think.cap.request.cons.chat.1
sys.heal.events.core.control.applied
svc.archive.events.mem.archiver.commit
```

During transition, older domains remain valid; CI will warn if no category prefix is present.

---

## SLO envelopes per category (for policy/Edge limits)
- **Consciousness:** P95 latency tight; throughput moderate; error budget low.
- **Subconsciousness:** latency moderate; throughput bursts tolerated; can backpressure.
- **Upper Cognition:** latency relaxed; parallelism high; cancellation on overload.
- **Memory:** durability first; latency flexible for writes; strict for read SLAs.
- **Nexus Core:** correctness & audit first; fan-out reliable.
- **Edge Anynode:** jitter minimal for ingress; strict policy evaluation path.

---

## Lifecycle (readiness gates)
- **Subconsciousness** online first (Hermes-Subcon).
- **Nexus Core** & **Edge** enabled next.
- **Consciousness** (front) after Core OK.
- **Memory** write path after Edge policy loaded.
- **Upper Cognition** gated by Viren + signed cutover.

This is the **authoritative taxonomy.** All services must declare a `category`, and all policy/routing will derive from it.
