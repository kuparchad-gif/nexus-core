# FDL Policy v0.1
Direct lane services callable by all agents:
- Procs (LLM): /v1/chat/completions (stream on)
- BERTs: /v1/embeddings, /v1/extract
- Memory: /v1/memory/*
Guardrails: budgets + lineage + data tags; audit to Loki; no policy gate.
