# Managed Model Liaison (MML)
**Every service and datastore is manned by an MML.**

- **Observe:** Ingest local events/log lines and write to Qdrant collection `mml_<service>`.
- **Reason:** Summarize or recommend using minimal local models or pluggable LLMs.
- **Act:** Propose signed intents (scale/purge/route.update) routed through **Lillith** by default.
- **No silent control:** Direct execution must be explicitly enabled per service and audited.

Collections (defaults):
- mml_consciousness, mml_subconsciousness, mml_memory, mml_edge, mml_monitor, mml_viren
- DB steward: **qdrant-mml**
