# Nexus Envelope Labels (Contract v1)

**Required**
- `tenant`: logical tenant (e.g., `acme`)
- `project`: project/app within tenant (e.g., `phoenix`)
- `service`: emitting module or microservice (e.g., `language`, `module-scribe`)
- `topic`: `logs|metadata|metrics|cold|audit|support|event`
- `privacy`: `internal|restricted|public`

**Recommended**
- `trace_id`: request or workflow UUID
- `env`: `dev|staging|prod`
- `region`: short code (e.g., `us-east-1`)
- `version`: semantic version of emitter
- `component`: finer-grain part (e.g., `ingester`, `planner`)
- `dewey`: coldstore classification (when applicable)

**Notes**
- Labels must be **flat, string key/value**. Booleans/numbers should be stringified when uncertain.
- Use **lowercase** keys with hyphen or underscore separators.
