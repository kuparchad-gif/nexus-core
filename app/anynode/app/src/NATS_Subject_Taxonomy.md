# NATS Subject Taxonomy (v1)

```
ten.<tenant>.proj.<project>.<domain>.<event>
```

- `<tenant>`: tenant slug (e.g., `acme`)
- `<project>`: project slug (e.g., `phoenix`)
- `<domain>`: `language|cognition|visual|memory|orchestrator|hermes|service`
- `<event>`: free-form event slug (e.g., `spawn`, `support`, `notify`, `ingest`)

## Examples
- `ten.acme.proj.phoenix.orchestrator.spawn`
- `ten.acme.proj.phoenix.viren.support`
- `ten.acme.proj.phoenix.memory.notify`
