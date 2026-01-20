# TRANSPARENCY

Every privileged change requires a signature and a human reason, and is written to a tamper-evident hash chain.

- **Controller:** Lillith (service: consciousness)
- **Executor:** nexus-core (ingress, DB, scraping)
- **Ledger:** /var/lib/nexus/audit/ledger.jsonl (mounted volume)

### Read
- `GET /audit/last`
- `GET /audit/ledger?limit=200`
- `GET /audit/verify`

If `CORE_AUDIT_READ_TOKEN` is set, pass: `x-core-audit-token: $TOKEN`.
