# Nexus Ports — Roster (current working set)

Service / Component           | Protocol | Port  | Status (yours) | Notes
------------------------------|----------|-------|----------------|-------------------------------
Qdrant (vector DB)            | HTTP     | 6333  | **UP**         | `GET /collections`
Qdrant (gRPC, optional)       | gRPC     | 6334  | (n/a)          | Only if exposed
SmokeEmbedder (FastAPI)       | HTTP     | 8016  | **UP**         | `POST /embed {"text":"ping"}`
Puppeteer Sidecar (HTTP)      | HTTP     | 8088  | (starting)     | `/health`, `/navigate`, `/screenshot`
Archiver (FastAPI)            | HTTP     | 9020  | planned        | `/health`, `/archive`, `/search`
Loki (forensics)              | HTTP     | 9011  | planned        | `/health`, `/log`
VIREN (healer)                | HTTP     | 9012  | planned        | `/health`, `/patch`
WireCrawler: Survey           | HTTP     | 9061  | planned        | `GET /survey?colony=...`
WireCrawler: Apply            | HTTP     | 9062  | planned        | `POST /apply`
(stdio) Puppeteer MCP server  | stdio    | —     | built          | spawned by sidecar (no port)

> Keep stdio servers (like Puppeteer MCP) behind the Sidecar; only expose the Sidecar’s HTTP port.
