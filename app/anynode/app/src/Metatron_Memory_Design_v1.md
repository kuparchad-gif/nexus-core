# Metatron Memory — Design v1 (Beyond Cache)

Memory is a lattice (inspired by Metatron's cube). We connect experiences along facets
(temporal, causal, agentic, modal, topical, spatial) and rank retrieval by composite geodesics
instead of plain nearest vectors.

Objects:
- Atom {id, ts, modality, tags, value, vec}
- Capsule {id, atoms[], synopsis, salience, permanence, lineage_id, tags}
- Lattice with facet-weighted edges.

Permanence P = ws*Survival + wh*Human + wt*Truth + wr*Recency + wu*Usage (weights sum to 1).

Pipeline:
1) Ingest → Atoms + facet edges.
2) Estimate salience → candidate Capsules.
3) Promote if P > τ; write-through to Archiver; backlinks maintained.
4) Age with decay; demote cold items.

Retrieval:
- Seed by vector search; expand by facet geodesics (beam); return Capsules + rationale.
