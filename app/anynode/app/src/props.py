from __future__ import annotations
from math import pi, sqrt
import cmath

C = 299_792_458.0
MU0 = 4.0 * pi * 1e-7
EPS0 = 1.0 / (MU0 * C * C)

MEDIA_DEFAULTS = {
    "EARTH": {"eps_r": 5.5,   "sigma": 1e-6,  "mu_r": 1.0},
    "AIR":   {"eps_r": 1.0006,"sigma": 1e-14, "mu_r": 1.0},
    "FIRE":  {"eps_r": 1.0,   "sigma": 1e-4,  "mu_r": 1.0},
    "WATER": {"eps_r": 80.0,  "sigma": 5e-3,  "mu_r": 1.0},
}

def lossy_em_props(eps_r: float, mu_r: float, sigma: float, f_hz: float):
    w = 2.0 * pi * f_hz
    mu = MU0 * mu_r
    eps = EPS0 * eps_r
    eps_c = complex(eps, -sigma / w)
    gamma = 1j * w * cmath.sqrt(mu * eps_c)
    alpha = gamma.real
    beta  = gamma.imag
    vp    = w / beta if beta != 0 else float("nan")
    delta = 1.0 / alpha if alpha > 0 else float("inf")
    eta   = cmath.sqrt(1j * w * mu / (sigma + 1j * w * eps))
    return {"alpha": alpha, "beta": beta, "vp": vp, "delta": delta, "eta_abs": abs(eta)}

def elemental_em(medium: str, f_hz: float, meters: float = 1.0, overrides: dict | None = None):
    from math import exp
    m = MEDIA_DEFAULTS.get(medium.upper())
    if m is None: raise ValueError("Unknown medium (EARTH|AIR|FIRE|WATER)")
    if overrides: m = {**m, **overrides}
    p = lossy_em_props(m["eps_r"], m["mu_r"], m["sigma"], f_hz)
    atten = exp(-p["alpha"] * meters)
    z_scale = 377.0 / p["eta_abs"] if p["eta_abs"] else 1.0
    phase = p["beta"] * meters
    return {"atten": atten, "z_scale": z_scale, "phase": phase, **p}
