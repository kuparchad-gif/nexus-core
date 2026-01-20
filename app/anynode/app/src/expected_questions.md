# Expected Questions (and where to point)
- Who can change routes? → Only Lillith (signed), see `__core/control/routes` and ledger proof.
- What if the ledger is corrupted? → `/audit/verify` fails; changes halt (fail-closed).
- How do you purge personal data? → `/__core/control/purge` flow; receipt; negative lookup procedure.
- How do you rotate keys over 30y? → `schedule/key_ceremony.yaml` + *_FILE secret support + crypto agility note.
- What stops misuse by the owner? → Public read window on audit + refusal policy in code.
