# Door State (Read-Only by Default)

- **discover** (no Viren check): read-only probes (OpenAPI, OIDC, GraphQL introspection, robots, sitemap).  
- **handshake** (Viren consulted): request permission to prepare scopes/keys; still no writes.  
- **enter** (Viren **must** allow): only then is active integration permitted (writes/tools).

The Scout never writes to targets. It only packages artifacts to **Memory Gateway** for vectorization.


# Capability Cards (integration)

Each artifact discovered (OpenAPI/OIDC/GraphQL/etc.) is turned into a **capability card**:
- Envelope v1 (`topic=integration`), with labels: `tenant`, `project`, `service`, `target_host`, `capability`.
- Blob: the raw doc (openapi.json, oidc config, graphql schema).

Others can query via vectors (Memory→Archiver) without touching the target system.
