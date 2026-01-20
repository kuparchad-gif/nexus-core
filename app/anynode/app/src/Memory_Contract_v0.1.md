# Memory Contract v0.1
Hot/working store + semantic index; write-through to Archiver; read-through on miss.

API:
- POST /v1/memory/put {key,value,tags,ttl,sensitivity}
- POST /v1/memory/get {key}
- POST /v1/memory/search {query,top_k,filters?,scope?}
- POST /v1/memory/capsule.create {source,content,lineage_id,retention,tags,...weights}
- POST /v1/memory/capsule.get {id}
- POST /v1/memory/evict {key|capsule_id}

Data controls: data_tags, redaction, retention, lineage_id, budgets (always attached).
