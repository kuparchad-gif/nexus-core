# Gateway v1 (OpenAI-compatible)

**Listen:** :8787 (container port 8787)

## Modes
- GATEWAY_MODE=CloudPrimary|LocalOnly|Offline
- CLOUD_BASE_URL / CLOUD_API_KEY
- LOCAL_BASE_URL / LOCAL_API_KEY / LOCAL_MODEL

## Endpoints
- POST /v1/chat/completions
- POST /v1/completions
- POST /v1/embeddings
- POST /v1/images/generations
- POST /v1/audio/transcriptions
- GET  /v1/models
- POST /v1/tools/run

## Telemetry
- Cannon → http://metanet:3100 (X-Scope-OrgID: fake)
- Envelope: {signal_id, origin, band, subject, context{user,trace,policies}, payload{…}}

## Compose
- Path: C:\Projects\Nexus_Meta_Hermes\NEXUS\NexusClusters\clusters\gateway\compose.yaml
- Launcher: C:\Projects\Nexus_Meta_Hermes\NEXUS\NexusClusters\clusters\gateway\Invoke-Gateway.ps1  (Build|Up|Down|Logs|Status|EnableAutostart|DisableAutostart)
