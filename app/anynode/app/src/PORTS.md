# Nexus Ports (v3)

| Service | Port |
|---|---:|
| Pineal (intent) | 7011 |
| Stem (router) | 7012 |
| WS Bus (reserved) | 7070 |
| Loki (external) | [REDACTED-URL] |

**Default admin:** $(admin)  
[REDACTED-SECRET-LINE]
Rotate via: POST /admin/rotate on Stem (Basic auth required). Rotation writes new hash to .env.local.

