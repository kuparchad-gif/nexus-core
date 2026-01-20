# Consent & Revocation
- Processing requires purpose-bound, revocable consent.
- Revocation endpoint: `/sovereignty/revoke` (POST).
- Propagation SLA: purge must complete within defined windows (see `procedures/revocation_SOP.md`).
- Proof: tombstone receipt hash appears in audit ledger; memory returns negative lookup thereafter.
