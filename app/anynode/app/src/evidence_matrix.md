# Evidence Matrix (Where to Look)
| Claim | Evidence | How to Verify |
|---|---|---|
| No silent control | Control endpoints require signature + reason | Run `scripts/prove_routes.sh` without reason (expect 400), then with reason (200) |
| Tamper-evident audit | `/audit/verify` is true; last entry chained | Run `scripts/audit_tail.sh` |
| Revocation works | Purge request → receipt + negative lookup | Run `scripts/prove_purge.sh` |
| Scalable pools | Signed scale request recorded in ledger | Run `scripts/prove_scale.sh` |
