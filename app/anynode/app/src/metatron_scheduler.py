import math
import torch
from torch import nn

class VortexScheduler(nn.Module):
    def __init__(self, num_layers: int, device=None, beta_base: float = 0.5):
        super().__init__()
        self.phi = (1 + 5 ** 0.5) / 2.0
        self.num_layers = num_layers
        self.register_buffer("beat", torch.tensor([1,2,4,8,7,5,1,2,4], dtype=torch.float32))
        self.mu = nn.Parameter(torch.zeros(num_layers))
        self.beta = nn.Parameter(torch.full((num_layers,), beta_base))

    def forward(self, step: int):
        t = torch.tensor(float(step), device=self.mu.device)
        s = self.phi * torch.sin(2 * math.pi * t / 9.0)
        idx = torch.arange(self.num_layers, device=self.mu.device) % 9
        m = self.beat.to(self.mu.device)[idx]
        gates = torch.sigmoid(self.mu + self.beta * s * m)
        return gates
