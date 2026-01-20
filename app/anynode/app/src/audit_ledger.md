# Audit Ledger Policy
- Medium: JSON Lines, append-only, hash-chained.
- Location: `/var/lib/nexus/audit/ledger.jsonl` (volume).
- Integrity: `/audit/verify` recomputes chain.
- Access: read-only HTTP with optional token; sensitive fields redacted.
- Rotation: snapshot quarterly to WORM storage.
