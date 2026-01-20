# NATS Subject Conventions

- `mem.shard.<shard_id>.remember`       : write memory events
- `mem.shard.<shard_id>.context.req`    : context request
- `mem.shard.<shard_id>.context.resp`   : context response
- `mem.enriched.events`                 : enriched frames from runtime
- `mesh.shard.register`                 : shard announces presence
- `mesh.shard.heartbeat`                : periodic heartbeat
- `audit.mem.events`                    : journaled memory events
- `bin.nim.frames`                      : NIM frames (transport)
- `bus.errors`                          : error channel
