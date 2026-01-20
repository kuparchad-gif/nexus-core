# SOP: Pool Scale
1. Identify target (`anynode` or `acidemikube`), pool/group, replicas.
2. Draft human reason (evidence: latency, RPS, queue depth).
3. Lillith signs `intent.pool.scale`.
4. Core forwards to manager; audit receipt written.
5. Validate SLOs; scale back when reason expires or TTL reached.
