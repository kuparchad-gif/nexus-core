# horn_scheduler.py — Reference Horn-shaped scheduler
import math
from typing import List, Tuple

def horn_rate(t: float, C: float = 1.0, k: float = 1.5) -> float:
    """Gabriel's Horn-shaped ramp toward capacity C.
    Fast initial rise, asymptotic taper. k controls steepness (>0).
    """
    return C * (1.0 - 1.0 / (1.0 + k * t))

def plan_chunks(total_tokens: int, steps: int = 8) -> List[int]:
    """Allocate progressive chunk sizes following the horn rate shape."""
    C = float(total_tokens)
    plan = []
    prev = 0.0
    for i in range(1, steps + 1):
        t = i / steps
        curr = horn_rate(t, C=C, k=1.8)
        plan.append(max(1, int(round(curr - prev))))
        prev = curr
    # fix rounding
    delta = total_tokens - sum(plan)
    if delta != 0:
        plan[-1] += delta
    return plan

if __name__ == "__main__":
    print(plan_chunks(256, steps=8))
