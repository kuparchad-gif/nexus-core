# bench/diff_report.py — compare two bench outputs
import json, sys, statistics
if len(sys.argv) < 3:
    print("Usage: python bench/diff_report.py bench/baseline.json bench/chad.json")
    sys.exit(2)
a = json.load(open(sys.argv[1],"r",encoding="utf-8"))
b = json.load(open(sys.argv[2],"r",encoding="utf-8"))
def tps(rows): return [r["toks_per_sec"] for r in rows if r.get("toks_per_sec")]
a_m, b_m = statistics.median(tps(a["rows"])), statistics.median(tps(b["rows"]))
speedup = (b_m / a_m) if a_m else 0
print(f"Baseline p50: {a_m:.2f} tok/s")
print(f"Chad p50   : {b_m:.2f} tok/s")
print(f"Speedup    : {speedup:.2f}x")
if speedup >= 2.0:
    print("✅ Meets the 2.00× throughput criterion.")
else:
    print("❌ Falls short of 2.00×.")
