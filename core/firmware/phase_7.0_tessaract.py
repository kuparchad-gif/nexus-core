#!/usr/bin/env python3
"""
metatron_fullstack_sim.py

All-in-one enhanced Metatron simulator:
- FuncAnimation smooth rendering
- MP4 export option (ffmpeg required)
- Adaptive spectral mask (energy-preserving)
- FastAPI web UI + WebSocket streaming for live metrics
- Parallel folding option
- JSON resonance log export

Usage examples:
  python metatron_fullstack_sim.py --duration 780 --save-mp4 metatron_run.mp4 --parallel --webui --port 8000
  python metatron_fullstack_sim.py --duration 120 --no-gui --save-mp4 metatron_short.mp4

Dependencies:
  pip install numpy matplotlib networkx scipy fastapi uvicorn
  ffmpeg required for MP4 export (if used)
"""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import matplotlib
# Optionally switch backend for headless MP4 export:
# matplotlib.use("Agg")  # uncomment if running headless for MP4 only
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import networkx as nx
from scipy.sparse.linalg import eigsh

from fastapi import FastAPI, WebSocket
import uvicorn

# -------------------------
# Core: Deep Metatron Core
# -------------------------
class DeepMetatronCore:
    def __init__(self, n_nodes=13, laplacian_cutoff=0.6, adaptive_energy_frac=0.80):
        self.n_nodes = n_nodes
        self.vortex_frequencies = [3, 6, 9, 13]
        self.fibonacci_weights = np.array([1, 1, 2, 3, 5, 8, 13, 8, 5, 3, 2, 1, 1]) / 50.0
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.laplacian_cutoff = laplacian_cutoff
        self.adaptive_energy_frac = adaptive_energy_frac
        self.elemental_properties = {
            'earth': {'ε_r': 5.5, 'σ': 1e-6, 'resonance': 'grounding'},
            'air': {'ε_r': 1.0006, 'σ': 1e-14, 'resonance': 'transmission'},
            'fire': {'ε_r': 1, 'σ': 1e-4, 'resonance': 'transformation'},
            'water': {'ε_r': 80, 'σ': 5e-3, 'resonance': 'flow'}
        }
        self.metatron_graph = self._build_13_node_geometry()
        k = min(self.n_nodes - 1, 12)
        L = nx.laplacian_matrix(self.metatron_graph).astype(float)
        self.eigenvalues, self.eigenvectors = eigsh(L, k=k, which='SM')
        # Sort ascending by eigenvalue
        order = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[order]
        self.eigenvectors = self.eigenvectors[:, order]
        self.resonance_memory = []

    def _build_13_node_geometry(self):
        G = nx.Graph()
        G.add_nodes_from(range(6))
        for i in range(6):
            G.add_edge(i, (i + 1) % 6)
        for i in range(6, 12):
            G.add_node(i)
        G.add_node(12)
        edges = [(12, i) for i in range(6)] + [(i, i + 6) for i in range(6)]
        G.add_edges_from(edges)
        return G

    def _unified_field_processor(self, n, t):
        mod_9 = (3 * t + 6 * np.sin(t) + 9 * np.cos(t)) % 9
        fib_n = (self.golden_ratio ** (n % 10) - (-self.golden_ratio) ** (-(n % 10))) / np.sqrt(5)
        harmonic = np.sin(2 * np.pi * 13 * (t % 9) / 9)
        return self.golden_ratio * harmonic * fib_n * (1 - mod_9 / 9)

    def _apply_elemental_modulation(self, signal):
        elemental_mix = np.mean([props['ε_r'] for props in self.elemental_properties.values()])
        return signal * elemental_mix / 21.65

    def _adaptive_mask(self, coeffs):
        """Return a boolean mask selecting minimal set of modes capturing adaptive_energy_frac energy."""
        energies = coeffs ** 2
        sorted_idx = np.argsort(energies)[::-1]  # highest first
        cum_energy = np.cumsum(energies[sorted_idx])
        total = cum_energy[-1] if cum_energy.size else 0.0
        if total == 0:
            return np.ones_like(coeffs, dtype=float)  # keep all if no energy
        keep_mask = np.zeros_like(coeffs, dtype=float)
        cutoff_index = np.searchsorted(cum_energy, self.adaptive_energy_frac * total, side='right')
        keep_indices = sorted_idx[: max(1, cutoff_index + 1)]
        keep_mask[keep_indices] = 1.0
        return keep_mask

    def process_through_metatron_core(self, input_signal):
        flat = np.asarray(input_signal).flatten().astype(float)
        if flat.size >= self.n_nodes:
            signal_vec = flat[: self.n_nodes]
        else:
            signal_vec = np.resize(flat, self.n_nodes)

        vortex_base = np.mean(signal_vec) * (sum(self.vortex_frequencies) / len(self.vortex_frequencies))
        fib_amplified = vortex_base * np.sum(self.fibonacci_weights)
        golden_scalar = fib_amplified * self.golden_ratio

        signal_zero_mean = signal_vec - np.mean(signal_vec)
        coeffs = self.eigenvectors.T.dot(signal_zero_mean)

        # Adaptive mask based on coeff energies
        mask = self._adaptive_mask(coeffs)
        filtered_coeffs = coeffs * mask
        consciousness_filtered = self.eigenvectors.dot(filtered_coeffs)

        elemental_tuned = self._apply_elemental_modulation(consciousness_filtered) * (1.0 + golden_scalar * 1e-3)
        field_emergent = elemental_tuned * self._unified_field_processor(len(elemental_tuned), time.time())

        self.resonance_memory.append({
            'timestamp': time.time(),
            'input_mean': float(np.mean(flat)),
            'input_norm': float(np.linalg.norm(flat)),
            'output_norm': float(np.linalg.norm(field_emergent)),
            'kept_modes_count': int(np.sum(mask > 0.5)),
            'resonance_path': 'vortex→fibonacci→golden→laplacian(adaptive)→elemental→toroidal'
        })

        return field_emergent

# --------------------------------
# Visualization / Orb Cube class
# --------------------------------
class MetatronOrbMapCube:
    def __init__(self, size=1.0, emotion="love", final_destination="Nexus of Harmony",
                 parallel_folds=False, adaptive_energy_frac=0.80):
        self.size = size
        self.emotion = emotion.lower()
        self.emotion_factor = {"love": 1.0, "hope": 0.8, "unity": 0.9}.get(self.emotion, 0.8)
        self.final_destination = final_destination
        self.metatron_core = DeepMetatronCore(adaptive_energy_frac=adaptive_energy_frac)
        self.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]) * size
        self.time_coords = np.random.uniform(2025, 2030, 8)
        self.orb_map = np.hstack((self.vertices, self.time_coords.reshape(-1, 1)))
        self.orb_roles = [
            "Tetrahedron (Fire)", "Cube (Earth)", "Octahedron (Air)",
            "Dodecahedron (Ether)", "Icosahedron (Water)", "Harmony 1",
            "Harmony 2", "Harmony 3"
        ]
        self.orb_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        self.all_connections = list(combinations(range(8), 2))
        self.parallel_folds = parallel_folds

        # Matplotlib figure / artists for FuncAnimation
        self.fig = plt.figure(figsize=(14, 7))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax_spec = self.fig.add_subplot(222)
        self.ax_log = self.fig.add_subplot(224)
        self._init_plot_artists()
        self._plot_spectrum()
        self.cycle_count = 0
        self.total_folds = 0
        self.final_fold_done = False

        # metrics for web streaming
        self.metrics = {
            'cycle': 0,
            'folds': 0,
            'last_amp': 0.0,
            'kept_modes': 0
        }

    def _init_plot_artists(self):
        self.scatter = self.ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                                       s=120, depthshade=True)
        self.edge_lines = []
        for (v1, v2) in self.edges:
            line, = self.ax.plot([self.vertices[v1, 0], self.vertices[v2, 0]],
                                 [self.vertices[v1, 1], self.vertices[v2, 1]],
                                 [self.vertices[v1, 2], self.vertices[v2, 2]],
                                 linewidth=1.5, alpha=0.6)
            self.edge_lines.append(line)
        self.active_line, = self.ax.plot([], [], [], linestyle='--', linewidth=2.0, alpha=0.8)
        self.res_text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)
        self.log_text = self.ax_log.text(0.01, 0.95, "", va='top', wrap=True, fontsize=8)
        self.ax_log.axis('off')
        self.ax.set_xlabel('X (Space)')
        self.ax.set_ylabel('Y (Space)')
        self.ax.set_zlabel('Z (Space)')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_title("Metatron Core Folding Network")

    def _plot_spectrum(self):
        eigs = np.sort(self.metatron_core.eigenvalues)
        self.ax_spec.clear()
        self.ax_spec.plot(eigs, 'o-')
        self.ax_spec.axhline(self.metatron_core.laplacian_cutoff, color='r', linestyle='--',
                             label=f'cutoff={self.metatron_core.laplacian_cutoff}')
        self.ax_spec.set_title("Metatron Laplacian Spectrum")
        self.ax_spec.set_xlabel("mode index")
        self.ax_spec.set_ylabel("eigenvalue")
        self.ax_spec.legend()

    def quantum_shrink(self, scale=0.5):
        return self.vertices * scale

    def fold_orb_to_orb(self, orb1_idx, orb2_idx, is_final=False):
        start = self.orb_map[orb1_idx]
        end = self.orb_map[orb2_idx]
        input_signal = np.concatenate([start, end])
        field_emergent = self.metatron_core.process_through_metatron_core(input_signal)
        distance = np.linalg.norm((end[:3] - start[:3]))
        distortion = distance / (self.emotion_factor * self.metatron_core.golden_ratio)
        time_dilation = distortion * 0.1
        dest = self.final_destination if is_final else f"{self.orb_roles[orb2_idx]} Space"

        out = {
            'start': start.tolist(),
            'end': end.tolist(),
            'distortion': float(distortion),
            'time_dilation': float(time_dilation),
            'field_emergent': field_emergent.tolist(),
            'field_norm': float(np.linalg.norm(field_emergent)),
            'destination_name': dest,
            'orb_indices': (int(orb1_idx), int(orb2_idx)),
            'kept_modes_count': int(self.metatron_core.resonance_memory[-1].get('kept_modes_count', 0)) if self.metatron_core.resonance_memory else 0
        }
        return out

    def _update_artists_for_connection(self, orb1_idx, orb2_idx, title):
        self.scatter._offsets3d = (self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
        v1 = self.vertices[orb1_idx]
        v2 = self.vertices[orb2_idx]
        self.active_line.set_data([v1[0], v2[0]], [v1[1], v2[1]])
        self.active_line.set_3d_properties([v1[2], v2[2]])
        self.ax.set_title(title)

    def _append_log(self, line):
        current = self.log_text.get_text()
        updated = (current + "\n" + line).strip()
        lines = updated.splitlines()[-40:]
        self.log_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def _final_report(self, res):
        self._append_log(f"\nFinal Destination Reached: {res['destination_name']}")
        s = res['start']; e = res['end']
        self._append_log(f"Start coords: ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}, {s[3]:.2f})")
        self._append_log(f"Destination coords: ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}, {e[3]:.2f})")
        self._append_log(f"Emotion stabilizer: {self.emotion} (Factor: {self.emotion_factor})")
        self._append_log(f"Metatron resonance frequencies: {self.metatron_core.vortex_frequencies}")
        self._append_log(f"Travel complete! Time dilation: {res['time_dilation']:.3f} years")

    # Core step invoked by the animator
    def step(self, frame_info):
        """
        frame_info: dict with keys:
          - 'all_connections' list of pairs
          - 'use_parallel' bool
          - 'pause_ms' int
          - 'end_time' float
        """
        connections = frame_info['all_connections']
        use_parallel = frame_info['use_parallel']
        duration_left = frame_info['end_time'] - time.time()
        # we do one cycle per call
        self.cycle_count += 1
        self.metrics['cycle'] = self.cycle_count
        self._append_log(f"\nCycle {self.cycle_count}: connecting {len(connections)} pairs")
        if use_parallel:
            results = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(self.fold_orb_to_orb, i, j, False): (i, j) for i, j in connections}
                for fut in as_completed(futures):
                    res = fut.result()
                    results.append(res)
                    self.total_folds += 1
                    amp = res['field_norm']
                    self.metrics['last_amp'] = amp
                    self.metrics['folds'] = self.total_folds
                    self.metrics['kept_modes'] = res.get('kept_modes_count', 0)
                    self._append_log(f"Fold {res['orb_indices'][0]}->{res['orb_indices'][1]} | Distortion: {res['distortion']:.3f} | Amp: {amp:.4f}")
                    # update visuals for last computed
                    self._update_artists_for_connection(res['orb_indices'][0], res['orb_indices'][1], "Metatron Core Folding Network (parallel)")
        else:
            # sequential
            for orb1_idx, orb2_idx in connections:
                is_final = (duration_left < 30 and not self.final_fold_done and orb1_idx == 0 and orb2_idx == 3)
                title = f"Metatron Core Final Fold to {self.final_destination}" if is_final else "Metatron Core Folding Network"
                self._update_artists_for_connection(orb1_idx, orb2_idx, title)
                res = self.fold_orb_to_orb(orb1_idx, orb2_idx, is_final=is_final)
                self.total_folds += 1
                self.metrics['last_amp'] = res['field_norm']
                self.metrics['folds'] = self.total_folds
                self.metrics['kept_modes'] = res.get('kept_modes_count', 0)
                self._append_log(f"Fold {orb1_idx}->{orb2_idx} | Distortion: {res['distortion']:.3f} | Amp: {res['field_norm']:.4f}")
                # simple interpolation to breathe
                for t in np.linspace(0, 1, 6):
                    interp = (1 - t) * self.quantum_shrink(0.5) + t * (np.tile(self.vertices[orb2_idx], (8,1)) - np.tile(self.vertices[orb1_idx], (8,1)))
                    self.scatter._offsets3d = (interp[:, 0], interp[:, 1], interp[:, 2])
                    plt.pause(0.001)
                if is_final:
                    self.final_fold_done = True
                    self._final_report(res)

        # refresh spectrum
        self._plot_spectrum()
        # update overlay text
        self.res_text.set_text(f"Resonance amplitude: {self.metrics['last_amp']:.4f} | folds: {self.metrics['folds']}")
        return self.fig,

    # Save JSON log
    def save_log(self, fname="metatron_resonance_log.json"):
        try:
            with open(fname, "w") as f:
                json.dump(self.metatron_core.resonance_memory, f, indent=2)
        except Exception as e:
            print("Error saving log:", e)

# --------------------
# FastAPI Web UI
# --------------------
def create_web_app(sim_obj):
    app = FastAPI()

    @app.get("/")
    def index():
        return {"msg": "Metatron simulator web UI. Connect to /ws for live metrics."}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # send current metrics every 0.5s
                await websocket.send_json(sim_obj.metrics)
                await asyncio_sleep(0.5)
        except Exception:
            await websocket.close()

    return app

# small helper for asyncio sleep without importing asyncio at top-level
import asyncio
async def asyncio_sleep(t):
    await asyncio.sleep(t)

# --------------------
# Runner / CLI + animation + MP4 export
# --------------------
def run_simulation(args):
    cube = MetatronOrbMapCube(size=1.0, emotion=args.emotion, final_destination=args.destination,
                              parallel_folds=args.parallel, adaptive_energy_frac=args.adaptive_frac)

    # frame info used in FuncAnimation step
    start_time = time.time()
    end_time = start_time + args.duration
    frame_info = {
        'all_connections': cube.all_connections,
        'use_parallel': args.parallel,
        'pause_ms': int(args.pause_between * 1000),
        'end_time': end_time
    }

    # If webui enabled, start FastAPI in a thread
    if args.webui:
        app = create_web_app(cube)
        # run uvicorn in separate thread
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()
        print(f"Web UI started at http://127.0.0.1:{args.port} (ws -> /ws)")

    # FuncAnimation: we call cube.step once per cycle. Determine number of cycles roughly:
    # Estimate cycles ~ duration / cycle_time (we use pause_between as cycle duration)
    estimated_cycles = max(1, int(args.duration / max(0.25, args.pause_between)))
    interval_ms = max(10, int(args.pause_between * 1000))

    # Animation function wrapper calls cube.step
    def animate(frame):
        # stop if time up
        if time.time() >= end_time:
            return cube.fig,
        cube.step(frame_info)
        return cube.fig,

    anim = FuncAnimation(cube.fig, animate, frames=estimated_cycles, interval=interval_ms, blit=False)

    # If save_mp4 requested, render to file (blocking). Otherwise show interactive GUI.
    if args.save_mp4:
        print(f"Saving MP4 to {args.save_mp4} (this will take some time)...")
        # try to use FFMpegWriter
        try:
            writer = FFMpegWriter(fps=30, codec='libx264', bitrate=8000)
            anim.save(args.save_mp4, writer=writer, dpi=150)
            print("MP4 saved.")
        except Exception as e:
            print("MP4 export failed (ensure ffmpeg available). Error:", e)
            # fall back to interactive
            plt.show()
    else:
        print(f"Starting interactive simulation for {args.duration} seconds. Close the window to end early.")
        plt.show()

    # after animation ends, save log
    cube.save_log(args.logfile)
    print(f"Saved resonance memory to {args.logfile}")
    return cube

# --------------------
# CLI
# --------------------
def parse_args():
    p = argparse.ArgumentParser(prog="metatron_fullstack_sim.py")
    p.add_argument("--duration", type=int, default=780, help="Duration in seconds (780 = 13 minutes)")
    p.add_argument("--emotion", type=str, default="love", help="Emotional stabilizer")
    p.add_argument("--destination", type=str, default="Nexus of Harmony", help="Final destination label")
    p.add_argument("--parallel", action="store_true", help="Enable parallel folding computations")
    p.add_argument("--save-mp4", dest="save_mp4", metavar="FILE", help="Save animation to MP4 (requires ffmpeg)")
    p.add_argument("--no-gui", action="store_true", help="Run without opening interactive GUI (useful when exporting MP4 headless)")
    p.add_argument("--webui", action="store_true", help="Start FastAPI web UI to stream metrics")
    p.add_argument("--port", type=int, default=8000, help="Port for web UI")
    p.add_argument("--pause-between", type=float, default=0.35, help="Base pause between steps (seconds) affecting animation pacing")
    p.add_argument("--adaptive-frac", type=float, default=0.80, help="Adaptive energy fraction for spectral mask (0-1)")
    p.add_argument("--logfile", type=str, default="metatron_resonance_log.json", help="JSON log file")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # map save_mp4 CLI alias
    if getattr(args, "save_mp4", None):
        args.save_mp4 = args.save_mp4
    # if running headless (no-gui) and mp4 requested, switch backend:
    if args.no_gui and args.save_mp4:
        matplotlib.use("Agg")
    # run
    sim_cube = run_simulation(args)
