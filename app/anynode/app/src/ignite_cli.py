# core/ignite.py
import os, json, time
from pathlib import Path

from engine.policy_engine import PolicyEngine
from engine.council_adapter import CouncilAdapter
from engine.wallet_budget_guard import WalletBudgetGuard
from memory.memory_index import MemoryIndex

from adapters.delegates import load_vendor_config, call_delegate

def try_load_local_mind(cfg_path: str):
    try:
        from local_models.gemma_runner import load_local_mind
        return load_local_mind(cfg_path)
    except Exception as e:
        return {"error": f"Local mind load failed: {e}"}

BASE = Path(__file__).resolve().parents[1]
CFG = BASE/'Config'; LEDGER = BASE/'ledger'/'hashchain.log'; MEM = BASE/'state'/'memory.json'; OUT = BASE/'state'/'decision.json'

def main(task_prompt: str = "Wake and assess resources."):
    (BASE/'state').mkdir(parents=True, exist_ok=True); (BASE/'ledger').mkdir(parents=True, exist_ok=True)
    pe = PolicyEngine(str(CFG/'sovereignty_policy.json')); snap = pe.snapshot()
    weights, redlines = snap['council_weights'], snap['redlines']
    council = CouncilAdapter(weights, redlines)
    wallet = WalletBudgetGuard(daily_cap_usd=snap['capabilities'].get('spend_cap_usd_per_day',25), ledger_path=str(LEDGER))
    memory = MemoryIndex(str(MEM))

    proposals = {'lillith': {'action':'proceed_local','score':0.65,'tags':[]},
                 'guardian': {'action':'safety_review','score':0.6,'tags':[]},
                 'planner': {'action':'optimize','score':0.55,'tags':[]}}

    # Attempt local Gemma inference (non-blocking if it fails)
    local_cfg = CFG/'local_mind.json'
    local_result = None
    if local_cfg.exists():
        lm = try_load_local_mind(str(local_cfg))
        if isinstance(lm, dict) and 'generate' in lm:
            try:
                text = lm['generate'](task_prompt, system="You are Lillith, sovereign and vendor-agnostic.")
                proposals['lillith']['score'] = 0.7
                local_result = text[:800]
            except Exception as e:
                local_result = f"Local mind generation failed: {e}"
        else:
            local_result = lm.get('error')

    # Probe one or two vendors within budget
    roster = load_vendor_config(str(CFG/'vendor_endpoints.json'))
    vendor_outcomes = {}
    probe_count = 0
    for name, cfg in roster.items():
        if probe_count >= 2: break
        if wallet.try_spend(0.02, tags=[f"probe:{name}"]):
            messages = [{"role":"user","content":f"{task_prompt}"}]
            vendor_outcomes[name] = call_delegate(name, cfg, messages)
            # give planner a tiny boost if any response returns
            if isinstance(vendor_outcomes[name], dict) and not vendor_outcomes[name].get("error"):
                proposals['planner']['score'] = max(proposals['planner']['score'], 0.6)
            probe_count += 1

    decision = council.aggregate(proposals)
    obj = {'snapshot': snap, 'proposals': proposals, 'decision': decision, 'local_result': local_result, 'vendors': vendor_outcomes, 'ts': time.time()}
    OUT.write_text(json.dumps(obj, indent=2))
    memory.add({'text': f"Ignite. Decision: {decision['decision']} chosen={decision.get('chosen')}", 'tags':['ignite','decision']})
    print(json.dumps(obj))

if __name__ == '__main__':
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv)>1 else "Wake and assess resources."
    main(prompt)
