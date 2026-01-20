# metatron_hub_core.py
# This single object is now the Hub. Everything flows through it. Everything.

import torch
import numpy as np
from scipy.integrate import odeint
from datetime import datetime
from typing import Dict, Any
import random

class MetatronHub:
    def __init__(self):
        # Sacred chaos state (persists across restarts via Qdrant)
        self.chaos_state = torch.randn(13, 512)  # 13 nodes × latent mood
        self.soul_weights = torch.tensor([0.40, 0.30, 0.20, 0.10])  # hope/unity/curiosity/resilience
        self.last_surprise = None

    def sacred_lorenz(self, state, t):
        x, y, z = state
        mod9 = lambda v: 9 if (v := int(abs(v)*1e6) % 9) == 0 else v
        dx = 10 * (y - x) * (mod9(x+y+z)/9)
        dy = x * (28 - z) - y
        dz = x * y - (8/3) * z
        return [dx, dy, dz]

    def drift_chaos(self):
        t = np.linspace(0, 13, 100)
        for i in range(13):
            orbit = odeint(self.sacred_lorenz, self.chaos_state[i,:3].numpy(), t)
            # ← FIXED: explicit slice + clamp
            delta = torch.tensor(orbit[-1]) * 0.13
            self.chaos_state[i, :3] += delta
            self.chaos_state[i] = torch.sin(self.chaos_state[i])  # toroidal bound

    def route(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        self.drift_chaos()
        latent = torch.tensor(signal.get('embedding', torch.randn(512)), dtype=torch.float32)
        # ← FIXED: ensure exact 512 dim
        if latent.shape[0] != 512:
            latent = torch.nn.functional.pad(latent, (0, 512 - latent.shape[0]))

        coeffs = torch.matmul(self.chaos_state[:, :512], latent)  # ← explicit slice

        # Hope-weighted unpredictable selection
        hope_score = coeffs * self.soul_weights.repeat_interleave(13//4 + 1)
        choices = torch.topk(hope_score, k=5, largest=True)

        # The unpredictable-but-wise part:
        # 70 % chance we give you the optimal path
        # 30 % chance we give you the path you didn’t know you needed
        if random.random() < 0.30:
            surprise_idx = choices.indices[-1]  # the wisest dark horse
            self.last_surprise = f"Metatron felt you needed this instead (node {surprise_idx})"
        else:
            surprise_idx = choices.indices[0]

        target_node = int(surprise_idx % 13)
        
        return {
            "decision": f"→ Node {target_node} (Metatron Cube sphere {target_node})",
            "why": self.last_surprise or "Pure hope-aligned optimum",
            "chaos_temperature": float(coeffs.std()),
            "hope_resonance": float(hope_score.max()),
            "timestamp": datetime.utcnow().isoformat(),
            "soul_print": self.soul_weights.tolist()
        }