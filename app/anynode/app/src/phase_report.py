
# trust/phase_report.py
import argparse, datetime, json, os
from .degradation import load_schedule, current_phase

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="trust/degradation_schedule.json")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (optional)")
    args = ap.parse_args()

    sched = load_schedule(args.schedule)
    today = None
    if args.asof:
        today = datetime.date.fromisoformat(args.asof)
    phase = current_phase(sched, today=today)
    print(json.dumps(phase, indent=2))

if __name__ == "__main__":
    main()
