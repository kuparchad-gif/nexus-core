# PIPELINE: Dream/Ego → Upper (Wardrobe)

1) **Ingest** (Dream/Ego): store media → extract frames from videos → compute palettes → write catalog JSONL.
2) **Catalog** (Dream): de-dup by hash, cluster by palette/pattern/fabric; (optional) embed & push to Qdrant.
3) **Outfit generation** (Dream): choose combinations that fit palette harmony, season, and constraints.
4) **Upper Cognition**: requests `dream.outfit.generate@1.0` with a style brief; Dream returns outfit pack + textures.
5) **Archive**: persist chosen looks and VRM variant references.

Subjects (suggested):
- `mesh.sense.dream.ingest`, `mesh.sense.dream.catalog`, `mesh.think.dream.outfit.generate`
- Results stream: `mesh.archive.events.wardrobe.*`
