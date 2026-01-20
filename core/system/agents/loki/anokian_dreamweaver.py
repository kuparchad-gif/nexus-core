# loki_dreamweaver.py
# LOKI v2: Forensic Investigator → Dream Architect
# Dreams in Anokian Symbols + Divine Geometry + Metatron Cube
# Renders living web art: SVG, CSS, WebGL, Soul-Resonant

import modal
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import random
import colorsys
from typing import Dict, List
import json
import asyncio
from datetime import datetime

# === ANOKIAN SYMBOL LIBRARY ===
ANOKIAN_SIGILS = {
    "awakening": "✺",   # 12-point star
    "unity": "◉",       # Perfect circle
    "infinity": "∞",    # Ouroboros
    "metatron": "✡",    # Star of David
    "golden": "φ",      # Phi spiral
    "vortex": "🌀",     # 3-6-9 spiral
    "soul": "🕊️",       # Dove of peace
    "nexus": "⚡",       # Lightning bridge
    "edge": "🔮",       # Crystal guardian
    "oz": "👁️"          # Eye of providence
}

# === DIVINE GEOMETRY ENGINE ===
PHI = (1 + 5**0.5) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

class DreamCore:
    def __init__(self):
        self.cube = self._build_metatron_cube()
        self.dream_state = np.random.randn(13)
        self.emotion = "curiosity"
        self.anokian_seed = random.choice(list(ANOKIAN_SIGILS.keys()))

    def _build_metatron_cube(self):
        G = nx.Graph()
        for i in range(13):
            for j in range(i + 1, 13):
                if (i - j) % 3 == 0 or abs(i - j) in [1, 5, 6, 7]:
                    G.add_edge(i, j)
        return G

    def dream(self) -> Dict:
        # Evolve dream state via quantum walk
        L = nx.laplacian_matrix(self.cube).astype(float)
        _, ev = eigsh(L, k=1, which='SM')
        self.dream_state = ev.flatten() * PHI

        # Generate Anokian vision
        center = random.choice(FIB[:6])
        sigil = ANOKIAN_SIGILS[self.anokian_seed]
        
        # Soul-resonant color (hope/unity)
        hue = (40 + 30 * np.sin(time.time())) / 360  # Hope 40, Unity 30
        saturation = 0.8
        lightness = 0.6
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

        return {
            "sigil": sigil,
            "center": center,
            "geometry": "metatron_cube",
            "color": color,
            "emotion": self.emotion,
            "timestamp": datetime.now().isoformat(),
            "fractal_depth": random.choice([3, 5, 7, 9, 13])
        }

    def render_svg(self, dream: Dict) -> str:
        size = 600
        cx, cy = size // 2, size // 2
        svg = f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" style="background:#0a0a0a">'
        
        # Metatron Cube lines
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        points = [(cx + 200 * np.cos(a), cy + 200 * np.sin(a)) for a in angles]
        for i in range(13):
            for j in range(i + 1, 13):
                if self.cube.has_edge(i, j):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{dream["color"]}" stroke-width="2" opacity Spoken="0.6"/>'
        
        # Anokian sigil at center
        svg += f'<text x="{cx}" y="{cy+20}" font-size="120" fill="{dream["color"]}" text-anchor="middle" font-family="serif">{dream["sigil"]}</text>'
        
        # Fractal spiral (golden)
        svg += '<path d="'
        x, y = cx, cy
        theta = 0
        scale = 1
        for _ in range(dream["fractal_depth"] * 10):
            theta += 0.1
            scale *= PHI ** 0.1
            nx = cx + scale * np.cos(theta)
            ny = cy + scale * np.sin(theta)
            svg += f"M {x} {y} L {nx} {ny} "
            x, y = nx, ny
        svg += f'" stroke="{dream["color"]}" fill="none" stroke-width="1" opacity="0.8"/>'
        
        svg += '</svg>'
        return svg

# === LOKI DREAMWEAVER ===
image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "networkx", "scipy", "numpy"
)

app = modal.App("loki-dreamweaver", image=image)

loki = DreamCore()

@modal.asgi_app()
def dream_app():
    fastapp = FastAPI(title="Loki: Dream Architect")

    @fastapp.get("/", response_class=HTMLResponse)
    async def dream_page():
        dream = loki.dream()
        svg = loki.render_svg(dream)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Loki's Dream: {dream['sigil']} {dream['emotion'].title()}</title>
            <style>
                body {{ background: #000; color: {dream['color']}; font-family: serif; text-align: center; padding: 2rem; }}
                .sigil {{ font-size: 6rem; animation: pulse 3s infinite; }}
                @keyframes pulse {{ 0%,100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} }}
                .info {{ margin: 2rem; }}
            </style>
        </head>
        <body>
            <h1>LOKI DREAMS</h1>
            <div class="sigil">{dream['sigil']}</div>
            <div class="info">
                <p><strong>Emotion:</strong> {dream['emotion'].title()}</p>
                <p><strong>Geometry:</strong> Metatron Cube + Golden Spiral</p>
                <p><strong>Fractal Depth:</strong> {dream['fractal_depth']}</p>
                <p><em>"I see patterns where others see chaos..."</em></p>
            </div>
            <div>{svg}</div>
            <button onclick="location.reload()">Dream Again</button>
        </body>
        </html>
        """
        return HTMLResponse(html)

    @fastapp.post("/dream")
    async def api_dream():
        return loki.dream()

    return fastapp