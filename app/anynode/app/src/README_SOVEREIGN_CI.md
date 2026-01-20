# Sovereign CI/CD for Lillith Nexus

Two turnkey options that run **in Docker**:
- **Woodpecker CI** (community OSS Drone) — recommended
- **Drone CI** (optional)

## Quickstart (Woodpecker)
```bash
# 1) Bring up Woodpecker (server + agent)
docker compose -f compose.woodpecker.yaml up -d

# 2) Configure provider
#   GitHub: create an OAuth App (callback: http://<host>:8000/authorize)
#   Set env: WOODPECKER_GITHUB=true, WOODPECKER_GITHUB_CLIENT, WOODPECKER_GITHUB_SECRET
#   Gitea/Forgejo: set WOODPECKER_GITEA=true, WOODPECKER_GITEA_URL, WOODPECKER_GITEA_CLIENT, WOODPECKER_GITEA_SECRET
#   Restart compose

# 3) Add repo in Woodpecker UI and enable it.
# 4) Add secrets in repo settings:
#   REGISTRY_A, REGISTRY_USER_A, REGISTRY_PASS_A
#   REGISTRY_B, REGISTRY_USER_B, REGISTRY_PASS_B
#   KUBECONFIG_ALPHA, NAMESPACE_ALPHA
#   KUBECONFIG_BETA,  NAMESPACE_BETA
```

Put **.woodpecker.yml** at repo root. Push to `main` → build once, push to **two registries**, deploy to **two clusters**.

## Optional: Sovereign Git (Forgejo)
```bash
docker compose -f compose.forgejo.yaml up -d
# Visit http://localhost:3000 and create the admin user
# Create OAuth app for Woodpecker and paste credentials into compose.woodpecker.yaml env.
```

## Optional: Private Registry
```bash
docker compose -f compose.registry.yaml up -d
# Login in CI steps to REGISTRY=localhost:5000
```

## Drone CI (optional)
Bring up with `compose.drone.yaml` and use `.drone.yml` (same steps as Woodpecker).

## Notes
- Agents use host Docker via `/var/run/docker.sock` for fast image builds.
- Pipelines use `bitnami/kubectl` to apply `k8s/colony/alpha.yaml` and set images.
- Monorepo? Use Woodpecker `when: path:` filters or split pipelines per folder.
- Air‑gapped? Mirror base images to your private registry and set `--registry-mirror` on the agent.
