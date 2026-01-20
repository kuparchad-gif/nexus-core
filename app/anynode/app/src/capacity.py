from math import log2
def shannon_capacity(bandwidth_hz: float, snr_linear: float, eff: float = 0.9) -> float:
    return bandwidth_hz * log2(1.0 + max(0.0, snr_linear)) * eff
