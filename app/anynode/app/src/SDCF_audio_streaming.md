# SDCF — Streaming & Data Control Frames (Audio)

Transport: WebSocket (recommended) or HTTP chunked. One JSON object per line.

## Frame types
- content: { "type":"content", "format":"pcm16|opus|text", "data":"<bytes|text_base64>", "seq": N }
- control: { "type":"control", "op":"rate|pause|resume|handoff", "args":{...} }
- tool_call: { "type":"tool_call", "name":"...", "args":{...}, "call_id":"..." }
- tool_result: { "type":"tool_result", "call_id":"...", "ok":true, "result":{...} }
- policy: { "type":"policy", "data_tags":["pii","ops"], "redaction":"mask|drop", "retention":"hot|archive" }
- final: { "type":"final", "status":"ok|fail", "receipts":[...], "next_steps":[...] }

## Data Controls (send with the first control frame and update as needed)
{
  "type":"policy",
  "lineage_id":"<uuid>",
  "guard": { "rate_limit": 48000, "budget_tokens": 2000, "allow_tools": ["diagnose","search"] },
  "data_tags": ["ops"],
  "redaction": "mask",
  "retention": "hot"
}
