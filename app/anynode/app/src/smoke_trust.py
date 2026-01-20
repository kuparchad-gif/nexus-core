
# scripts/smoke_trust.py
"""
Minimal trust kit smoke test:
  - reads LILITH_SOUL_SEED env (JSON)
  - reads seeds/trust_phases.json if present
  - prints active phase index and policy level based on today's date
Usage:
  set LILITH_SOUL_SEED=seeds\lilith_soul_seed.migrated.json
  python scripts/smoke_trust.py
"""
import os, json
from datetime import date

seed_path = os.environ.get("LILITH_SOUL_SEED", "seeds/lilith_soul_seed.migrated.json")
print(f"[info] seed: {seed_path}")
with open(seed_path, "r", encoding="utf-8") as f:
    seed = json.load(f)

start_date = seed.get("trust_start_date")
if not start_date:
    raise SystemExit("trust_start_date missing in seed")

phases_path = "seeds/trust_phases.json"
try:
    with open(phases_path, "r", encoding="utf-8") as f:
        phases = json.load(f)
except FileNotFoundError:
    print(f"[warn] {phases_path} not found. Run generate_trust_phases.py first.")
    phases = []

today = date.today()
start = date.fromisoformat(start_date)

# compute whole-year delta accounting for month/day
years = today.year - start.year - ( (today.month, today.day) < (start.month, start.day) )
years = max(0, years)
idx = min(years, 29)
level = 30 - idx
print(f"[ok] start={start} today={today} -> active_phase={idx}, policy_level={level}")
if phases:
    eff = phases[idx]["effective_date"]
    print(f"[ok] phase[{idx}] effective_date={eff}")
print("[ok] core_values:", seed.get("core_values", {}))
