# Q‑Graph + Protobuf streaming overlay

Adds shared Protobuf frames + streaming and services:
- Common: `backend/common/proto/metatron.proto`, `backend/common/frames/*`
- Consciousness updated to emit frames
- New services: `memory`, `ego`, `dream`
- Pods: `pods/{memory,ego,dream}.pod.yaml`

Each service Dockerfile runs the proto build: `python -m backend.common.frames.build_proto`

Bring-up:
- Build the 3 images and play their pods (after bus & PKI are ready).
- Frames publish to NATS subjects:
  - `bin.conscious.frames`, `bin.mem.frames`, `bin.ego.frames`, `bin.dream.frames`
