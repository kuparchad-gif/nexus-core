
# scripts/generate_trust_phases.py
"""
Generate a 30-phase (30-year) degrade schedule from a start date.
Each phase has a policy_level (30 -> 1) and a short description.
Usage:
  python scripts/generate_trust_phases.py --start 2025-01-01 --out seeds/trust_phases.json
"""
import argparse, json
from datetime import date

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="seeds/trust_phases.json")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    phases = []
    for year in range(0, 30):
        phases.append({
            "phase": year,
            "effective_date": date(start.year + year, start.month, start.day).isoformat(),
            "policy_level": 30 - year,  # 30 down to 1
            "notes": "Auto-generated guardrail step (lower = looser)"
        })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(phases, f, indent=2)
    print(f"[ok] wrote {args.out} with {len(phases)} phases (start={start})")

if __name__ == "__main__":
    main()
