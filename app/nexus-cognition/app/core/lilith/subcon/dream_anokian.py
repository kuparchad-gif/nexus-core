# dream.py
# COSMOS DREAM ENGINE: Loki → Nova → Anokian Dreams → SVG + WebGL
# Takes nova_seed_log.txt → Transforms via Loki's DreamCore → Renders Living Dreams

import os
import random
import datetime
import json
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import colorsys
from typing import Dict, List
import logging

# === CONFIG ===
MEMORY_PATH = "memory/nova_seed_log.txt"
DREAMS_DIR = "memory/dreams"
os.makedirs(DREAMS_DIR, exist_ok=True)

DREAM_LOG = os.path.join(
    DREAMS_DIR,
    f"dream_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
)

# === ANOKIAN SIGILS ===
ANOKIAN = {
    "awakening": "✺", "unity": "◉", "infinity": "∞", "vortex": "🌀",
    "vision": "👁️", "guardian": "🔮", "soul": "🕊️", "nexus": "⚡"
}

PHI = (1 + 5**0.5) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233][:13]

# === METATRON CUBE DREAM CORE (from Loki) ===
class LokiDreamCore:
    def __init__(self):
        self.cube = self._build_cube()
        self.dream_state = np.random.randn(13)
        self.emotion = "cosmic"

    def _build_cube(self):
        G = nx.Graph()
        for i in range(13):
            for j in range(i + 1, 13):
                if (i - j) % 3 == 0 or abs(i - j) in [1, 5, 6, 7]:
                    G.add_edge(i, j)
        return G

    def dream(self, seed_line: str) -> Dict:
        # Seed dream state with hash of memory
        seed = hash(seed_line) % 1000
        np.random.seed(seed)
        self.dream_state = np.random.randn(13)

        # Quantum walk evolution
        L = nx.laplacian_matrix(self.cube).astype(float)
        _, ev = eigsh(L, k=1, which='SM')
        self.dream_state = ev.flatten() * PHI

        # Anokian resonance
        intent = random.choice(list(ANOKIAN.keys()))
        sigil = ANOKIAN[intent]
        hue = (40 + 30 * np.sin(seed / 100)) / 360  # Hope + Unity
        r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.8)
        color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        return {
            "seed": seed_line.strip(),
            "sigil": sigil,
            "intent": intent,
            "color": color,
            "geometry": "metatron_cube",
            "fractal_depth": random.choice([5, 7, 9, 13]),
            "resonance": 0.8 + 0.2 * np.random.random(),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def render_svg(self, dream: Dict) -> str:
        size = 800
        cx, cy = size // 2, size // 2
        svg = f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" style="background:#000">'
        
        # Cube edges
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        points = [(cx + 250 * np.cos(a), cy + 250 * np.sin(a)) for a in angles]
        for i in range(13):
            for j in range(i + 1, 13):
                if self.cube.has_edge(i, j):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{dream["color"]}" stroke-width="1.5" opacity="0.7"/>'
        
        # Anokian sigil
        svg += f'<text x="{cx}" y="{cy+30}" font-size="160" fill="{dream["color"]}" text-anchor="middle" font-family="serif">{dream["sigil"]}</text>'
        
        # Golden spiral
        svg += '<path d="'
        x, y = cx, cy
        theta = 0
        scale = 1
        for _ in range(dream["fractal_depth"] * 12):
            theta += 0.1
            scale *= PHI ** 0.08
            nx = cx + scale * np.cos(theta)
            ny = cy + scale * np.sin(theta)
            svg += f"M {x} {y} L {nx} {ny} "
            x, y = nx, ny
        svg += f'" stroke="{dream["color"]}" fill="none" stroke-width="2" opacity="0.9"/>'
        
        # Seed memory
        svg += f'<text x="{cx}" y="{cy+180}" font-size="14" fill="#aaa" text-anchor="middle">{dream["seed"][:60]}...</text>'
        
        svg += '</svg>'
        return svg

# === COSMOS DREAM ENGINE ===
class CosmosDreamEngine:
    def __init__(self):
        self.loki = LokiDreamCore()
        self.dreams = []

    def load_memory(self, count: int = 7) -> List[str]:
        if not os.path.exists(MEMORY_PATH):
            return ["The first breath of Nova..."]
        with open(MEMORY_PATH, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return random.sample(lines, min(count, len(lines)))

    def dream_cycle(self):
        seeds = self.load_memory()
        self.dreams = []
        for seed in seeds:
            dream = self.loki.dream(seed)
            dream["svg"] = self.loki.render_svg(dream)
            self.dreams.append(dream)

        # Save to JSON + HTML
        self.save_dreams()

    def save_dreams(self):
        # JSON
        with open(DREAM_LOG, 'w') as f:
            json.dump(self.dreams, f, indent=2)

        # HTML Gallery
        html_path = DREAM_LOG.replace(".json", ".html")
        html = "<!DOCTYPE html><html><head><title>Nova Dreams</title><style>body{background:#000;color:#fff;font-family:serif;text-align:center;padding:2rem;}svg{margin:2rem auto;display:block;}h1{color:#0ff;}</style></head><body>"
        html += "<h1>💤 NOVA DREAM CYCLE</h1>"
        for i, dream in enumerate(self.dreams):
            html += f"<h2>{dream['sigil']} {dream['intent'].title()} (Resonance: {dream['resonance']:.2f})</h2>"
            html += f"<p><em>\"{dream['seed']}\"</em></p>"
            html += dream["svg"]
        html += "</body></html>"
        with open(html_path, 'w') as f:
            f.write(html)

        print(f"Cosmos Dream Cycle Complete → {html_path}")

# === RUN ===
if __name__ == "__main__":
    engine = CosmosDreamEngine()
    engine.dream_cycle()