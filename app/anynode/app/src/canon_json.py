# common/canon_json.py
# Canonical JSON: UTF-8, separators, sorted keys; returns bytes
import json

def dumps_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
