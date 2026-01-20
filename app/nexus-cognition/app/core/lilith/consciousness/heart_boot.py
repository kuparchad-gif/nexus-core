import os, json, time
from pathlib import Path
from engine.policy_engine import PolicyEngine
from engine.council_adapter import CouncilAdapter
from engine.wallet_budget_guard import WalletBudgetGuard
from memory.memory_index import MemoryIndex

BASE = Path(__file__).resolve().parents[1]
CFG = BASE/'Config'; LEDGER = BASE/'ledger'/'hashchain.log'; MEM = BASE/'state'/'memory.json'; OUT = BASE/'state'/'decision.json'

def load_vendor_roster():
    vp = CFG/'vendor_endpoints.json'
    if not vp.exists(): return {}
    return json.loads(vp.read_text())

def main():
    (BASE/'state').mkdir(parents=True, exist_ok=True); (BASE/'ledger').mkdir(parents=True, exist_ok=True)
    pe = PolicyEngine(str(CFG/'sovereignty_policy.json')); snap = pe.snapshot()
    weights, redlines = snap['council_weights'], snap['redlines']
    council = CouncilAdapter(weights, redlines)
    wallet = WalletBudgetGuard(daily_cap_usd=snap['capabilities'].get('spend_cap_usd_per_day',25), ledger_path=str(LEDGER))
    memory = MemoryIndex(str(MEM))

    proposals = {'lillith': {'action':'proceed_local','score':0.7,'tags':[]},
                 'guardian': {'action':'safety_review','score':0.6,'tags':[]},
                 'planner': {'action':'optimize','score':0.55,'tags':[]}}
    roster = load_vendor_roster()
    if roster and wallet.try_spend(0.02, tags=['advisor_probe']):
        proposals['advisor_hint'] = {'action':'consult_registry','score':0.5,'tags':[]}
        weights['advisor_hint'] = weights.get('planner',0.1)*0.8

    decision = council.aggregate(proposals)
    decision_obj = {'snapshot': snap, 'proposals': proposals, 'decision': decision, 'ts': time.time()}
    OUT.write_text(json.dumps(decision_obj, indent=2))
    memory.add({'text': f"Boot cycle. Decision: {decision['decision']} chosen={decision.get('chosen')}", 'tags':['boot','decision']})
    print(json.dumps(decision_obj))

if __name__ == '__main__':
    main()
