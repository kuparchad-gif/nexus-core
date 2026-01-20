# Rules of Engagement (ROE)
- Do not ingest or export real user data.
- Use only the provided **synthetic PII** test set for purge demos.
- All privileged actions must include a **human reason**; otherwise expect HTTP 400.
- Use signing secret as provided in `secrets/core_control_secret` (or as configured).
- All test actions become public in the sandbox ledger; assume persistent storage.
