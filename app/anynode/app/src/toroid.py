from __future__ import annotations
import math
PHI = (1 + 5 ** 0.5) / 2.0
def _fib(n: int) -> float:
    rt5 = 5 ** 0.5
    phi = PHI
    psi = (1 - rt5) / 2.0
    return (phi**n - psi**n) / rt5
def unified_toroid(n: int, t: float) -> float:
    mod9 = (3.0 * t + 6.0 * math.sin(t) + 9.0 * math.cos(t)) % 9.0
    harmonic = math.sin(2.0 * math.pi * 13.0 * t / 9.0)
    return PHI * harmonic * _fib(max(0, n)) * (1.0 - mod9 / 9.0)
def apply_toroid_gain(vec: list[float], phase_level: int, t: float, gain: float = 0.05) -> list[float]:
    g = unified_toroid(phase_level, t)
    scale = 1.0 + gain * (g / PHI)
    return [x * scale for x in vec]
