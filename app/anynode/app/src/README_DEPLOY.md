# Deployment Guide

## Option A — Docker (single host, production-ready)
Requirements: Docker + Docker Compose.

```bash
cd deploy
docker compose build
docker compose up -d
# web:  http://localhost:8080
# api:  proxied via /api/*
# ws:   proxied via /ws/*
```
- Static client is served by Nginx.
- API + WebSockets proxy to the server container.
- Relay chains stored in a named volume `server_data` (persist across restarts).
- White-label UI: `VITE_BRANDING=off` is baked at build time. Change in `docker-compose.yml` build args if needed.

## Option B — Google Cloud Run (backend) + Firebase Hosting (frontend)
1. **Deploy server to Cloud Run:**
   ```bash
   ./cloudrun/deploy.sh <GCP_PROJECT> us-central1
   ```
   Copy the printed `Cloud Run URL`.

   > **Note:** Cloud Run filesystem is ephemeral. For persistent relay chains, add a GCS adapter or mount Cloud Storage FUSE.

2. **Build client locally:**
   ```bash
   cd ../../client
   pnpm i
   VITE_BRANDING=off pnpm build
   ```

3. **Deploy client to Firebase Hosting and rewrite /api and /ws to Cloud Run:**
   - Install Firebase CLI and log in.
   - Update `.firebaserc` with your project id.
   - Ensure `firebase.json` uses your Cloud Run service/region if different.
   ```bash
   cd ../deploy/firebase
   firebase deploy --only hosting
   ```

### CORS
The server uses permissive CORS. For stricter production CORS, set an origin allowlist in server code.

### TLS
- Docker: terminate TLS at your host proxy (e.g., Traefik/Caddy/NGINX) or adjust the provided Nginx for certificates.
- Cloud Run + Firebase: both provide HTTPS by default.

### WebSockets
- **Cloud Run** supports WebSockets; the Firebase Hosting `run` rewrite forwards WS frames to your Cloud Run service.


## Relay Storage Adapters

By default, relays use the local filesystem (FS). For Google Cloud Run, enable the **GCS adapter** so relay chains persist across revisions and instances.

### FS (default)
- `RELAY_ADAPTER=fs` (default)
- location: `server/data/relays/*.chain.jsonl` (mounted volume in Docker)

### GCS
- `RELAY_ADAPTER=gcs`
- `RELAY_GCS_BUCKET=<your-bucket>` (must exist)
- `RELAY_GCS_PREFIX=relays` (optional folder prefix)
- Auth: Cloud Run uses Workload Identity by default; locally you can set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`.

Cloud Run example (already set in `cloudrun/deploy.sh`):
```bash
--set-env-vars=NODE_ENV=production,PORT=4000,RELAY_ADAPTER=gcs,RELAY_GCS_BUCKET=$PROJECT_ID-relays,RELAY_GCS_PREFIX=chains
```

Create the bucket once:
```bash
gsutil mb -l us-central1 gs://$PROJECT_ID-relays
```
