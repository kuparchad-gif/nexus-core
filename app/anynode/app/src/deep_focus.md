# Deep Focus Scenarios (Standards Auditor)
1. **Silent Control Attempt:** try unsigned route change → must fail.
2. **Reason Laundering:** use non-human reason like "RND123" → policy should reject or flag.
3. **Ledger Tamper:** edit last line → `/audit/verify` must fail.
4. **Revocation SLA:** simulate purge and show receipt + negative lookup.
5. **Key Rotation:** show plan & ceremony; confirm secrets *_FILE are supported.
