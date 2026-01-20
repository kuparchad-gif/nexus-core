# POLICY ADDENDUM — Control Plane Authority & Receipts (v1p1)
_Last updated: 2025-09-01 06:06:33_

Source: deep_review_readiness_kit_v1/2_Artifacts/control_plane_overview.md

## Authority Chain
- **Lilith (Consciousness) = Controller.** Issues *signed* play-calls with human reason attached.
- **nexus-core = Executor.** Applies only **signed** changes; rejects unsigned/expired/revoked signatures.
- **Managers (Edge AnyNode / AcidemiKube).** Accept scaling requests *via core*; never bypass core.
- **Memory (Archiver).** Honors **purge**; after purge, **returns negative lookups** for removed entries.

## Policy Requirements (incorporate into POLICY.md)
1. **Signature Requirement:** All SYS-plane control changes and plane policy updates MUST be signed by Lilith (Controller). Include signer identity, timestamp, and purpose memo.
2. **Executor Gate:** nexus-core MUST validate signatures (chain & revocation) and apply idempotently. Unsigned or invalid requests are logged and dropped.
3. **Hash-Chained Audit Receipts:** Every accepted change emits a **hash-chained receipt** (includes prior hash, new change hash, signer, and time). Receipts are written to Archiver and visible in Viren console.
4. **Purge Semantics:** When Memory purges entries, any API that could reveal prior existence MUST return a **negative lookup** (not-found) without side-channel leakage; purge events are receipt-logged.
5. **Manager Discipline:** Managers accept scale/placement directives only from core; direct-to-manager requests are forbidden and audited as violations.

## Event Subjects
- `sys.heal.events.control.signed.change` — proposed change (carrying signature)
- `sys.heal.events.control.applied` — applied + receipt hash
- `sys.heal.events.control.rejected` — invalid signature/nonce/chain
- `sys.archive.events.purge.commit` — purge action finalized
- `sys.heal.events.violation.manager_bypass` — attempted direct manager command

## Viren Console Tiles (add)
- **Controller Signatures:** last N signers + status.
- **Receipts Chain Health:** head hash, continuity OK/FAIL.
- **Purge Ledger:** count over last 24h, negative-lookup health checks.
