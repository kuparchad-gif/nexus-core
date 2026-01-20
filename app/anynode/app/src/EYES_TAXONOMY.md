# Nexus Eyes — Canon (v1)
Updated: 2025-09-01 10:17:56

**What are the eyes of AI?** In Nexus: any ingress that lets the system *perceive* the world.

## Classes of eyes
- **Vision**: cameras (RGB/IR/depth), screen capture, video (RTSP/MP4).
- **Audio**: microphones, telephony, radio; ASR/diarization.
- **Text & Web**: documents, webpages, APIs, code, databases.
- **Sensors/IoT**: time-series (IMU, GPS, LiDAR, radar), sysmetrics.
- **User interaction**: prompts, clicks, cursor/keystrokes, game-engine ticks.
- **Self-perception**: logs/metrics/traces (Loki/Grafana) → *introspection eyes*.

## Where they live in Nexus
- **Edge Anynode (BBB)** → ingress + policy/consent + rate limits.
- **Sense mesh** (plane: svc|cog) → raw streams land here.
- **Subconsciousness** → codecs/filters (NIM, OCR, ASR, VLM adapters, PsyShield).
- **Memory** → durable logs, Q-Graph indices, retrieval.
- **Consciousness** → GH/Lilith consume condensed signals (e.g., 64D vectors) for awareness/reasoning.

## Subjects (canonical, category-prefixed)
- `svc.sense.cap.request.cons.vision.frame.1`   — RGB/IR/depth frames
- `svc.sense.cap.request.subc.audio.chunk.1`    — audio PCM chunks
- `cog.sense.cap.request.subc.ocr.text.1`       — OCR results
- `cog.sense.cap.request.subc.asr.text.1`       — ASR results
- `svc.sense.cap.request.subc.web.doc.1`        — fetched web/doc payloads
- `svc.sense.events.mem.log.entry`              — log entries (pre-Loki)
- `sys.heal.events.core.consent.update`         — consent/PII policy applied

> You can alias these to your existing domain keys (cons|subc|mem|core).

## Security/consent
- **Edge**: deny-by-default sources; per-source rate/PII masks; signed consent receipts.
- **PsyShield**: sanitize text before LLM/GH; detect manipulation & exfil.
- **Memory Sentinel**: veto risky commits (poisoning).
- **Aegis**: integrity on OS paths.

---
