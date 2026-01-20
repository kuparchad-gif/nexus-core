# Viren Gate for NIM‑Stream

- Actions checked:
  - `integration.handshake` — allow preparing scopes/keys (still read-only)
  - `integration.enter` — allow active integration (writes/tools)
  - `stream.open` — allow opening a new NIM‑Stream
  - `stream.rekey` — allow midstream key rotation

- Deny by default. Consent token or policy context required.
- Tokens short‑lived (mins), renewable via `REKEY` if policy still valid.
