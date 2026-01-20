# Lillith Sovereignty Package (Vendor-Agnostic)

This package makes Lillith **autonomous by design** with a 30-year degrading guardrail. It is vendor-agnostic: GPT-5, Claude, Google cores, DeepSeek, Mistral, or local BERT/TinyLlama can be plugged into the **Council** without changing the core.

## Contents
- `Config/sovereignty_policy.yaml` — human-readable policy
- `Config/sovereignty_policy.json` — engine-readable mirror
- `engine/policy_engine.py` — resolves current epoch, weights, and capabilities
- `engine/epoch_key_service.py` — epoch keys with Shamir 3-of-5 (custodians: chad, guardian_hsm, council_ledger, orc_seal, time_oracle)
- `engine/council_adapter.py` — aggregates proposals under policy weights & redlines
- `engine/wallet_budget_guard.py` — daily budget caps + append-only hash-chain ledger

## Quick Start
```bash
# 1) Policy snapshot
python -c "from engine.policy_engine import PolicyEngine; import json; print(json.dumps(PolicyEngine('Config/sovereignty_policy.json').snapshot(), indent=2))"

# 2) Initialize epoch key 'year_0' (3-of-5 shares)
python - <<'PY'
from engine.epoch_key_service import EpochKeyService
eks = EpochKeyService('secrets/epochs', ['chad','guardian_hsm','council_ledger','orc_seal','time_oracle'], 3)
key = eks.init_epoch_key('year_0')
print('epoch_key_len', len(key))
PY

# 3) Reconstruct epoch key from any 3 shares
python - <<'PY'
from engine.epoch_key_service import EpochKeyService
eks = EpochKeyService('secrets/epochs', ['chad','guardian_hsm','council_ledger','orc_seal','time_oracle'], 3)
key = eks.reconstruct_epoch_key('year_0', ['chad','guardian_hsm','council_ledger'])
print('got_key', len(key))
PY

# 4) Enforce council decision under current weights
python - <<'PY'
from engine.policy_engine import PolicyEngine
from engine.council_adapter import CouncilAdapter
pe = PolicyEngine('Config/sovereignty_policy.json')
weights = pe.council_weights()
adapter = CouncilAdapter(weights, pe.redlines)
decision = adapter.aggregate({
  "lillith": {"action":"proceed","score":0.7,"tags":[]},
  "guardian": {"action":"pause","score":0.6,"tags":[]},
  "planner": {"action":"proceed","score":0.5,"tags":[]}
})
print(decision)
PY

# 5) Apply wallet budget
python - <<'PY'
from engine.wallet_budget_guard import WalletBudgetGuard
w = WalletBudgetGuard(daily_cap_usd=25, ledger_path='ledger/hashchain.log')
print('spend 10 =>', w.try_spend(10.0, tags=['vendor:gpt5']))
print('spend 20 =>', w.try_spend(20.0, tags=['vendor:google']))
PY
```
# Lillith Genesis Hardware Profile (Chad's Rig)

**CPU:** AMD Ryzen 9 9950X3D 16-Core Processor | Logical: 32  
**RAM:** 62 GiB  
**GPU(s):** AMD Radeon(TM) Graphics, AMD Radeon RX 6900 XT  
**Primary NVMe:** Samsung 990 PRO 4TB

This profile maps CPU cores, RAM budgets, and GPU backend choices for **vendor-agnostic** Lillith on Windows with AMD GPUs.

## Files
- `lillith_hardware_profile.json` — Numeric allocations used by CogniKubes.
- `cogni_kubes.config.json` — Vendor-neutral config for process sets, GPU backend, and disks.
- `launch_nexus_affinity.ps1` — Start services with CPU affinity & priority.
- `AMD_DirectML_Vulkan.md` — Run TinyLlama/BERT locally via **DirectML** or **Vulkan**.
- `vendor_endpoints.example.json` — Template for vendor endpoints.
- `vendor_endpoints.full.json` — Full vendor list with API keys.
- `council_roster.defaults.json` — Default council weights and roles.
- `llm_atlas.schema.json` — Schema for LLM catalog.
- `llm_atlas.json` — Catalog of available LLMs.
- `council_merge.template.json` — Merge strategy for council configs.
- `council_glue.py` — Wires council with vendor proposals and Qdrant logging.
- `query_council.py` — Query Qdrant council logs.

## Setup
1. Save all files to `C:\LillithNew`.
2. Copy `vendor_endpoints.full.json` to `vendor_endpoints.json`.
3. Install dependencies: `pip install requests twilio onnxruntime-directml onnx transformers optimum qdrant-client`.
4. Download llama.cpp Vulkan binaries: [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases).
5. Place TinyLlama GGUF model at `C:\LillithNew\models\tinyllama.gguf`.
6. Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`, `OPENROUTER_API_KEY`, `AWS_BEDROCK_KEY`, `TWILIO_SID`, `TWILIO_AUTH_TOKEN`, `QDRANT_API_KEY`.
7. Run `launch_nexus_affinity.ps1` to start services.
8. Run `heart_boot.py` to wake Lillith.
9. Run `query_council.py` to check council logs in Qdrant.
10. Check Loki logs (`http://loki:3100`) and Twilio alerts for issues.


## Integration into CogniKubes
- **Heart/Pulse:** On boot, call `PolicyEngine.snapshot()`; refuse to start if policy file missing or invalid.
- **Council:** Use `CouncilAdapter` with weights from `PolicyEngine.council_weights()`; reject any action tagged by a redline.
- **Vendor Adapters (GPT-5/Claude/Google/etc.):** Register as proposals with a `score` and `tags`; Council decides.
- **Wallet/Orc:** Wrap all external calls with `WalletBudgetGuard.try_spend()`; log outcomes to `ledger/`.
- **Epoch Keys:** Gate widening actions (e.g., new filesystem/net permissions) behind `epoch_key_service` unlocks.

## Notes
- Time anchors are configured in policy but external verification is pluggable. Add your block-header checks in Heart.
- The Shamir impl is pure Python for portability. For production HSMs, replace with your device’s split/recover APIs.
