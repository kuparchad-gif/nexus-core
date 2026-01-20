# Registry Usage (No Rebuilds)

**Live directory (ephemeral):** NATS subjects + Redis
- Use NATS for broadcasts, fan-out, and signaling across tenants/projects.
- Cache fast lookups (active tickets, sessions, capability pointers) in Redis with keys:
  - `reg:<tenant>:<project>:svc:<service>:<kind>`

**Durable directory:** Memory → Archiver (vectorized envelopes)
- All envelopes (logs/metadata/metrics/cold) flow **via Memory first** and then to Archiver.
- Query vectors by label filters (`tenant`, `project`, `service`, `topic`) to discover assets, capabilities, incidents.

**Contract:** `docs/Envelope_v1.schema.json` and `Labels.md`
- Emitters: `archiver-client`, `module-scribe`, `coldstore-agent`, `viren-desk`, etc.
- Keep `labels` consistent so cross-tenant/project routing stays deterministic.
