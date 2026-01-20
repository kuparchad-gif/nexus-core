# Sovereign Mode

**Goal:** Operations and control decisions continue even if third-party AI providers change terms, revoke access, or Nova upgrades.

**Enforcements**
- Nova runs **offline by default** and only co-signs after lint.
- Core accepts **multiple Nova secrets** to permit rolling upgrades.
- All LLM usage is **local-first** with a global **egress kill switch** (`LLM_EGRESS_ALLOWED=false`).

**Drill:** Run `scripts/sovereign_drill.sh` and attach the ledger receipts to your evidence bundle.
