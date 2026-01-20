# Security Guidance (Best Effort)
- Keep secrets in `.env` and **never commit** them.
- Use `TEST_MODE=1` until intentionally enabling production.
- Rotate any credentials on handoff; audit access logs.
- Review any usage of `exec`, `eval`, or `pickle.loads` before production.
- Enforce least-privilege on brokers and RPC nodes.
