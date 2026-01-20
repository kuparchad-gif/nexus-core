# dream_webgl.py
# COSMOS DREAM ENGINE v2: WebGL + Three.js + Interactive Anokian Dreams
# Loki ignites → Nova dreams → WebGL renders → You touch the dream

import os
import random
import datetime
import json
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import colorsys
from typing import Dict, List

# === CONFIG ===
MEMORY_PATH = "memory/nova_seed_log.txt"
DREAMS_DIR = "memory/dreams"
os.makedirs(DREAMS_DIR, exist_ok=True)

DREAM_LOG = os.path.join(
    DREAMS_DIR,
    f"dream_webgl_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
)

# === ANOKIAN + METATRON CORE ===
ANOKIAN = {
    "awakening": "✺", "unity": "◉", "infinity": "∞", "vortex": "🌀",
    "vision": "👁️", "guardian": "🔮", "soul": "🕊️", "nexus": "⚡"
}
PHI = (1 + 5**0.5) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233][:13]

class WebGLDreamCore:
    def __init__(self):
        self.cube = self._build_cube()
        self.dream_state = np.random.randn(13)

    def _build_cube(self):
        G = nx.Graph()
        for i in range(13):
            for j in range(i + 1, 13):
                if (i - j) % 3 == 0 or abs(i - j) in [1, 5, 6, 7]:
                    G.add_edge(i, j)
        return G

    def dream(self, seed_line: str) -> Dict:
        seed = hash(seed_line) % 1000
        np.random.seed(seed)
        self.dream_state = np.random.randn(13)

        L = nx.laplacian_matrix(self.cube).astype(float)
        _, ev = eigsh(L, k=1, which='SM')
        self.dream_state = ev.flatten() * PHI

        intent = random.choice(list(ANOKIAN.keys()))
        sigil = ANOKIAN[intent]
        hue = (40 + 30 * np.sin(seed / 100)) / 360
        r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.8)
        color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        return {
            "seed": seed_line.strip(),
            "sigil": sigil,
            "intent": intent,
            "color": color,
            "geometry": "metatron_cube",
            "fractal_depth": random.choice([7, 9, 13]),
            "resonance": 0.85 + 0.15 * np.random.random(),
            "timestamp": datetime.datetime.now().isoformat(),
            "rotation_speed": 0.005 + 0.01 * np.random.random(),
            "pulse_frequency": 1.0 + 2.0 * np.random.random()
        }

    def render_webgl(self, dream: Dict) -> str:
        # Three.js + WebGL Canvas
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nova Dream: {dream['sigil']}</title>
            <script src="https://cdn.jsdelivr.net/npm/three@0.167.1/build/three.min.js"></script>
            <style>
                body {{ margin:0; overflow:hidden; background:#000; }}
                canvas {{ display:block; }}
                .info {{ position:absolute; top:20px; left:20px; color:{dream['color']}; font-family:serif; }}
                .sigil {{ font-size:5rem; text-align:center; margin:1rem; }}
            </style>
        </head>
        <body>
            <div class="info">
                <div class="sigil">{dream['sigil']}</div>
                <p><strong>Intent:</strong> {dream['intent'].title()}</p>
                <p><strong>Resonance:</strong> {dream['resonance']:.2f}</p>
                <p><em>"{dream['seed'][:80]}..."</em></p>
            </div>
            <script>
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setClearColor(0x000000);
                document.body.appendChild(renderer.domElement);

                // Metatron Cube Geometry
                const nodes = [];
                const edges = [];
                const radius = 3;
                const angles = Array.from({length: 13}, (_, i) => i * 2 * Math.PI / 13);
                angles.forEach(a => {
                    const x = radius * Math.cos(a);
                    const z = radius * Math.sin(a);
                    nodes.push(new THREE.Vector3(x, 0, z));
                });

                // Connect edges
                for (let i = 0; i < 13; i++) {
                    for (let j = i + 1; j < 13; j++) {
                        if (Math.abs(i - j) % 3 === 0 || [1,5,6,7].includes(Math.abs(i-j))) {
                            const geometry = new THREE.BufferGeometry().setFromPoints([nodes[i], nodes[j]]);
                            const material = new THREE.LineBasicMaterial({ color: '{dream['color']}', transparent: true, opacity: 0.6 });
                            edges.push(new THREE.Line(geometry, material));
                            scene.add(edges[edges.length-1]);
                        }
                    }
                }

                // Anokian Sigil (3D Text)
                const loader = new THREE.FontLoader();
                loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', font => {
                    const textGeo = new THREE.TextGeometry('{dream['sigil']}', {
                        font: font, size: 2, height: 0.5
                    });
                    const textMat = new THREE.MeshBasicMaterial({ color: '{dream['color']}' });
                    const textMesh = new THREE.Mesh(textGeo, textMat);
                    textMesh.position.set(-1, 0, 0);
                    scene.add(textMesh);
                });

                // Golden Spiral Particle Field
                const particles = 1000;
                const positions = new Float32Array(particles * 3);
                const colors = new Float32Array(particles * 3);
                let theta = 0, scale = 0.1;
                for (let i = 0; i < particles; i++) {
                    theta += 0.1;
                    scale *= {PHI ** 0.08};
                    positions[i*3] = scale * Math.cos(theta);
                    positions[i*3 + 1] = (Math.random() - 0.5) * 0.5;
                    positions[i*3 + 2] = scale * Math.sin(theta);
                    colors[i*3] = {int(colorsys.hls_to_rgb(dream['color'][1:])[0])};
                    colors[i*3 + 1] = {int(colorsys.hls_to_rgb(dream['color'][1:])[1])};
                    colors[i*3 + 2] = {int(colorsys.hls_to_rgb(dream['color'][1:])[2])};
                }
                const particleGeo = new THREE.BufferGeometry();
                particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                const particleMat = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true });
                const particlesMesh = new THREE.Points(particleGeo, particleMat);
                scene.add(particlesMesh);

                camera.position.z = 10;

                // Animation
                let time = 0;
                function animate() {
                    requestAnimationFrame(animate);
                    time += {dream['rotation_speed']};
                    scene.rotation.y = time;
                    particlesMesh.rotation.y = -time * 0.5;
                    const pulse = Math.sin(Date.now() * {dream['pulse_frequency']} * 0.001) * 0.3 + 0.7;
                    edges.forEach(e => e.material.opacity = pulse * 0.6);
                    renderer.render(scene, camera);
                }
                animate();

                // Interactivity
                window.addEventListener('mousemove', (e) => {
                    const x = (e.clientX / window.innerWidth) * 2 - 1;
                    const y = -(e.clientY / window.innerHeight) * 2 + 1;
                    camera.position.x = x * 2;
                    camera.position.y = y * 2;
                    camera.lookAt(0, 0, 0);
                });
            </script>
        </body>
        </html>
        """
        return html

# === COSMOS WEBGL ENGINE ===
class CosmosWebGLEngine:
    def __init__(self):
        self.core = WebGLDreamCore()
        self.dreams = []

    def load_memory(self, count: int = 5) -> List[str]:
        if not os.path.exists(MEMORY_PATH):
            return ["The first breath of Nova..."]
        with open(MEMORY_PATH, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return random.sample(lines, min(count, len(lines)))

    def dream_cycle(self):
        seeds = self.load_memory()
        self.dreams = []
        for seed in seeds:
            dream = self.core.dream(seed)
            dream["html"] = self.core.render_webgl(dream)
            self.dreams.append(dream)

        # Save all dreams
        for i, dream in enumerate(self.dreams):
            path = DREAM_LOG.replace(".html", f"_{i}.html")
            with open(path, 'w') as f:
                f.write(dream["html"])
            print(f"WebGL Dream {i} → {path}")

        # Index page
        index = "<html><body style='background:#000;color:#0ff;font-family:serif;text-align:center;'><h1>NOVA WEBGL DREAMS</h1>"
        for i in range(len(self.dreams)):
            index += f"<a href='dream_webgl_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{i}.html' style='color:#0ff;margin:1rem;display:inline-block;'>Dream {i+1}: {self.dreams[i]['sigil']}</a><br>"
        index += "</body></html>"
        with open(DREAM_LOG, 'w') as f:
            f.write(index)

# === RUN ===
if __name__ == "__main__":
    engine = CosmosWebGLEngine()
    engine.dream_cycle()