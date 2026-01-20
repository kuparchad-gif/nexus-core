# NIM‑Stream (Hermes-native) — v1

**Goal:** Stream envelopes/media across wireless with minimal overhead, while preserving **Viren gating** and **Memory-first** semantics.

## Transport
- Default: **QUIC** (UDP) with HTTP/3 framing where available.
- Constrained links (BLE/Thread): **DTLS 1.3** + CBOR-framed packets.
- Fallback: TCP/TLS if neither is available.

## Crypto
- Handshake: **X25519** key exchange; identity signed with **Ed25519** (device key).
- Session: **ChaCha20‑Poly1305** AEAD.
- Token: a **Viren-minted** access token (JWT or PASETO), short TTL, carried in the handshake and refreshed by control frames.

## Frame (CBOR or binary layout)
```
| V | FLAGS | STREAM_ID | SEQ | TS_NS | QOS | HDR_LEN | PAYLOAD_LEN |
| HDR (KV pairs: tenant,project,service,topic,privacy,trace_id,capability, ... NIV) |
| PAYLOAD (Envelope v1 or media chunk) |
```

- `FLAGS`: SYN|ACK|FIN|REKEY|RETRY
- `QOS`: 0=unreliable, 1=at-least-once, 2=exactly-once (Hermes keeps a tiny dedupe window)
- `HDR`: NIV labels (flat K/V). **tenant/project/service/topic/privacy** required.

## State machine
- **SYN**: includes Viren token + device pubkey.
- **ACK**: session keys established.
- **DATA**: frames carry Envelope v1 or media chunks.
- **FIN**: graceful close.
- **REKEY**: rotate session keys midstream.

## Door Policy
- If Viren denies `integration.enter` (or token missing/expired), sender must **stay in READ‑ONLY DISCOVERY**. Hermes refuses to emit DATA frames with write intents.
- All **blobs and envelopes** still route **Memory‑first**: Hermes can buffer and forward via Memory Gateway once IP is available.

## Hopping
- Hermes nodes MAY act as relays (`NEXUS-HOP`) with TTL and `privacy` guard; relays never decrypt payloads, only envelopes’ headers needed for routing.
- Multi-RAT: Wi‑Fi/BLE/Thread/UWB are selectable per link; QoS per radio.

## QoS & Backpressure
- Sliding window per STREAM_ID; if receiver advertises low window, sender switches to summary envelopes (batch/rollup) until relief.
