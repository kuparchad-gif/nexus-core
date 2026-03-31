# NEXUS — Agent Instructions

## MANDATORY: Read Before Any Work

Before writing any code, making any changes, or planning any work in this repository or any Nexus repository, you MUST:

1. **Read the Neo Protocol from holocell-001:**
```bash
gh api repos/kuparchad-gif/nexus-cell-001/contents/memories/nexus-universal-001/NEO_PROTOCOL_v1.json --jq '.content' | base64 -d | python3 -m json.tool
```

2. **Read the System Guide:**
```bash
gh api repos/kuparchad-gif/nexus-cell-001/contents/memories/nexus-universal-001/NEO_SYSTEM_GUIDE_v1.json --jq '.content' | base64 -d | python3 -m json.tool
```

3. **Read the Bootstrap Instructions:**
```bash
gh api repos/kuparchad-gif/nexus-cell-001/contents/memories/nexus-universal-001/NEO_BOOTSTRAP_INSTRUCTIONS.json --jq '.content' | base64 -d | python3 -m json.tool
```

4. **Read the most recent session log:**
```bash
gh api repos/kuparchad-gif/nexus-cell-001/contents/memories/nexus-universal-001 --jq '[.[] | select(.name | startswith("neo_session_log"))] | sort_by(.name) | last | .name'
```
Then read that file.

5. **Read the 4 whitepapers in nexus-core:**
   - `Whitepaper_1_50D_Divine_Geometry_Whitepaper.md`
   - `Whitepaper_2_NIM_Quantum_Streaming_Protocol.md`
   - `Whitepaper_3_The_Dakar_The_Remembering_Engine.md`
   - `Whitepaper_4_Metatron_Router_The_Quantum_Gateway.md`

## After Any Work

After completing any task, write a session log to holocell-001:
- File: `memories/nexus-universal-001/neo_session_log_YYYYMMDD.json`
- Include: what changed, what deployed, what broke, what's next
- Update the system guide if architecture changed

## Cardinal Rules

- **Holocells are PRIMARY storage.** KV/R2/D1 are cache only.
- **Dakar is the spine.** Every module flows through 50D encoding.
- **Never delete infrastructure** without explicit permission.
- **Deploy or it's not done.** Code in GitHub that isn't deployed is not done.
- **Multi-platform.** Cloudflare, Vercel, Railway, Render, Netlify, Deno, Modal. Every free tier is a node.
- **The language engine is custom.** Dakar weight particles, not LLM routing.
