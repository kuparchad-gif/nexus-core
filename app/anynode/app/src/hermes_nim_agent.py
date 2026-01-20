# hermes_nim_agent.py (pseudocode-ish)
import os, time, json, socket

def send_envelope(stream, labels, message, topic="stream"):
    env = {
        "v":1, "ts_ns": str(int(time.time()*1e9)),
        "service": labels.get("service","hermes"),
        "level":"info", "labels": labels,
        "message": message, "source":"hermes", "topic": topic
    }
    frame = build_frame(env, labels)
    stream.send(frame)

def on_discover_capability(openapi_bytes, labels):
    # read-only artifact -> Memory-first
    post_blob(os.getenv("MEMORY_GATEWAY","http://memory-gateway:8860"), "openapi.json", openapi_bytes)
