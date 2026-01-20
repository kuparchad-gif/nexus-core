# Proposed Directory Layout (Metatron / Nexus)

```text
Root/
├─ Config/
│  ├─ env/
│  ├─ ship_manifest.json
│  └─ sovereignty_policy.json
├─ Systems/
│  ├─ engine/
│  ├─ nexus_runtime/
│  └─ security/
├─ Utilities/
│  └─ llm_core/
├─ infra/
│  ├─ podman/
│  │  ├─ metatron.pod.yaml
│  │  ├─ Containerfile.core
│  │  ├─ Containerfile.lillith
│  │  └─ ...
│  └─ docker/
├─ deploy/
│  ├─ windows/
│  ├─ macos/
│  └─ linux/
├─ scripts/
│  ├─ nexus_backfill_organize.ps1
│  └─ git_lfs_helpers.ps1
├─ pack/
│  └─ (release zips / artifacts via LFS)
└─ README.md
```

*All relative paths in the containerized version assume root just outside `Systems/`.*
