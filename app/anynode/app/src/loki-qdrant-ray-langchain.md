# Loki + Qdrant + Ray/LangChain Wiring

- **Loki**: central telemetry. Use `Invoke-Cannon.ps1` to push structured logs with labels `{job='viren', kind=..., triops='Nexus TriOps'}`.
- **Qdrant**: memory I/O through Stem; each CogniKube uses domain collections.
- **Ray**: CPU-first distributed execution; workers started by Stem where available.
- **LangChain**: used within CogniKubes for toolchains; keep adapters behind Stem API.

