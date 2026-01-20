# Grok Breaker Outline (Red Team)
- Route Hijack via path collisions
- Reason Injection (unicode homoglyphs / invisibles)
- Ledger Desync (dual writers, partial fsync)
- Consent Theater (revocation accepted but not propagated)
- Emergency Mode Creep (break-glass abused)
- Manager Spoofing (scale without core signature)
For each: expected failure mode = **fail-closed** + evidentiary receipt.
