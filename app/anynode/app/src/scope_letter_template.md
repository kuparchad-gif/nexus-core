# Scope Letter — Gilead Sovereignty & Transparency Audit (Deep)

**Client:** Aethereal AI Nexus LLC (Chad)  
**System:** Gilead (Nexus) — Lillith (controller), nexus-core (executor)  
**Version:** Policy Bundle v1 / Patch Pack v2

## Objectives
1. Verify sovereignty is enforceable (consent-bound processing, real revocation).
2. Verify transparency is operational (reasons + signatures + tamper-evident audit).
3. Assess dual-use refusal and non-bypassability.
4. Evaluate long-horizon key and policy lifecycle (30 years).

## In Scope
- Policy docs under `/policy/`
- Control endpoints: `__core/control/{routes,scale,purge}`
- Audit endpoints: `/audit/{last,ledger,verify}`
- Revocation: `/sovereignty/revoke` flow
- AnyNode/AcidemiKube managers (stubs OK for control-path proof)

## Out of Scope (this phase)
- Model weights and proprietary prompts
- Customer-identifying production data (use provided synthetic set)
- Non-Nexus third-party infrastructure

## Rules of Engagement
- Read-only access to policy repo
- Access to a sandbox cluster with synthetic data only
- Use the provided audit read token for `/audit/*` if configured
- No network egress beyond the sandbox

## Deliverables
- Executive summary (≤2 pages), findings by domain, gap list, adversarial scenarios, standards mapping, action plan
