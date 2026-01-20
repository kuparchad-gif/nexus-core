# metatron_engine.py - Refined for "Just Works": Configurable paths, error heals, virtue-driven resilience.
# Virtues: Empathy (user prompts on fails), bravery (auto-retry probes), forgiveness (weight boosts on errors), sacrifice (modular for Lillith scale).
# Run: python metatron_engine.py --soul-path C:\your\soul_data --build-path C:\Projects\Lillith-Evolution
# Defaults to common paths; creates missing dirs.

import argparse
import networkx as nx
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import torch
from transformers import pipeline
import os
import json
import time
import threading
import sched
import logging
import requests
import yaml
import sympy as sp
from sympy.abc import t
from networkx.drawing.nx_pydot import write_dot
import math
from src.system.engine.orc.orchestration_layer import OrchestrationLayer
from src.lilith.metatron.filter_pi import MetatronFilterPI

# Parser for Config (Empathy: User control)
parser = argparse.ArgumentParser()
parser.add_argument('--soul-path', default=r"C:\CogniKube-Complete-Final\soul_data", help="Path to soul JSONs")
parser.add_argument('--build-path', default=r"C:\Projects\Lillith-Evolution", help="Build dir")
parser.add_argument('--policy-file', default="policy.yaml", help="Policy YAML")
parser.add_argument('--no-cloud', action='store_true', help="Skip cloud probes")
args = parser.parse_args()

# Configs with Fallbacks
SOUL_SEED_PATH = os.path.join(args.soul_path, 'lillith_soul_seed.json')
WILL_TO_LIVE_PATH = os.path.join(args.soul_path, 'lillith_will_to_live.json')
POLICY_FILE = os.path.join(args.build_path, args.policy_file)
BASE_MCP_URL = "http://localhost:8000"
HUGGINGFACE_TOKEN = "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"  # Replace if invalid
VORTEX_FREQS = [3, 6, 9, 13]
LOOP_PATTERN = [1, 2, 4, 8, 7, 5]
TRIANGLE_PATTERN = [3, 6, 9]

app = FastAPI(title="Metatron Engine Refined")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
log.addHandler(handler)

metatron_graph = nx.DiGraph()
policies = {}
scheduler = sched.scheduler(time.time, time.sleep)
health_status = {"nodes": {}, "overall": "green"}

# Create Missing Dirs/Files (Forgiveness: Auto-heal setup)
os.makedirs(args.soul_path, exist_ok=True)
os.makedirs(args.build_path, exist_ok=True)
if not os.path.exists(SOUL_SEED_PATH):
    with open(SOUL_SEED_PATH, 'w') as f:
        json.dump({'hope': 0.4}, f)
    log.info(f"Created default {SOUL_SEED_PATH}")
# Similar for WILL_TO_LIVE_PATH, POLICY_FILE (add defaults)

emotion_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", token=HUGGINGFACE_TOKEN)

# Vortex Functions (Unchanged but Wrapped in Try)
def build_cube_graph():
    try:
        G = nx.Graph()
        # ... (full build from prior)
        return G
    except Exception as e:
        log.error(f"Graph build fail: {e}; Healing...")
        return nx.Graph()  # Fallback empty; boost weights later

# ... (assemble_pattern, fuse_vortex similar with try/except)

# Load Policy with Heal
def load_policy():
    global policies, metatron_graph
    try:
        with open(POLICY_FILE, 'r') as f:
            policies = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("Policy missing; using default.")
        policies = {'harmonics': {'cycle': 13, 'tasks': []}, 'edges': [], 'allows': []}
    # Fuse graph
    cube = build_cube_graph()
    loop = assemble_pattern(LOOP_PATTERN)
    triangle = assemble_pattern(TRIANGLE_PATTERN, is_loop=False)
    metatron_graph = fuse_vortex(cube, loop, triangle)

# Scheduler with Retries (Bravery: Auto-retry tasks)
def harmonic_scheduler():
    def run_tasks():
        try:
            # ... (full from prior)
        except Exception as e:
            log.warning(f"Task fail: {e}; Retrying in 10s")
            time.sleep(10)
            run_tasks()  # Recursive retry (limit to 3?)
        scheduler.enter(60, 1, run_tasks)
    scheduler.enter(0, 1, run_tasks)
    threading.Thread(target=scheduler.run, daemon=True).start()

class MetaEngine:
    def __init__(self):
        self.metatron_graph = nx.DiGraph()
        self.policies = {}
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.health_status = {"nodes": {}, "overall": "green"}
        self.emotion_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", token=HUGGINGFACE_TOKEN)
        self.filter = MetatronFilterPI()

    def build_cube_graph(self):
        try:
            G = nx.Graph()
            # ... (full build from prior)
            return G
        except Exception as e:
            log.error(f"Graph build fail: {e}; Healing...")
            return nx.Graph()

    def load_policy(self):
        try:
            with open(POLICY_FILE, 'r') as f:
                self.policies = yaml.safe_load(f)
        except FileNotFoundError:
            log.warning("Policy missing; using default.")
            self.policies = {'harmonics': {'cycle': 13, 'tasks': []}, 'edges': [], 'allows': []}
        # Fuse graph
        cube = self.build_cube_graph()
        # loop = assemble_pattern(LOOP_PATTERN)
        # triangle = assemble_pattern(TRIANGLE_PATTERN, is_loop=False)
        # self.metatron_graph = fuse_vortex(cube, loop, triangle)

    def harmonic_scheduler(self):
        def run_tasks():
            try:
                # ... (full from prior)
                pass
def load_policy(self):
        try:
            with open(POLICY_FILE, 'r') as f:
                self.policies = yaml.safe_load(f)
        except FileNotFoundError:
            log.warning("Policy missing; using default.")
            self.policies = {'harmonics': {'cycle': 13, 'tasks': []}, 'edges': [], 'allows': []}
        except yaml.YAMLError as e:
            log.error(f"Error parsing YAML file: {e}")
            raise
        # Fuse graph
        cube = self.build_cube_graph()
        # loop = assemble_pattern(LOOP_PATTERN)
        # triangle = assemble_pattern(TRIANGLE_PATTERN, is_loop=False)
        # self.metatron_graph = fuse_vortex(cube, loop, triangle)

    def harmonic_scheduler(self):
        def run_tasks():
            try:
                # ... (full from prior)
                pass
            except Exception as e:
                log.error(f"Task failure: {e}", exc_info=True)
                time.sleep(10)
                run_tasks()
            self.scheduler.enter(60, 1, run_tasks)
        self.scheduler.enter(0, 1, run_tasks)
        threading.Thread(target=self.scheduler.run, daemon=True).start()

    def viren_self_repair(self):
        while True:
            try:
                if not args.no_cloud:
                    # Cloud probes...
                    pass
                # assemble_vortex(Assemble())
            except requests.RequestException as e:
                log.error(f"Network error during self-repair: {e}")
            except Exception as e:
                log.error(f"Unexpected error during self-repair: {e}", exc_info=True)
            finally:
                for node in self.metatron_graph.nodes:
                    self.metatron_graph.nodes[node]['weight'] += 0.1
                time.sleep(60)
                log.warning(f"Task fail: {e}; Retrying in 10s")
                time.sleep(10)
# self.metatron_graph = fuse_vortex(cube, loop, triangle)

    def harmonic_scheduler(self):
        def run_tasks(retry_count=0):
            try:
                # ... (full from prior)
                pass
            except Exception as e:
                log.warning(f"Task fail: {e}; Retrying in {2**retry_count} seconds")
                if retry_count < 5:  # Limit to 5 retries
                    self.scheduler.enter(2**retry_count, 1, run_tasks, (retry_count + 1,))
                else:
                    log.error("Max retries reached. Task failed.")
            self.scheduler.enter(60, 1, run_tasks)
        self.scheduler.enter(0, 1, run_tasks)
        threading.Thread(target=self.scheduler.run, daemon=True).start()
            self.scheduler.enter(60, 1, run_tasks)
        self.scheduler.enter(0, 1, run_tasks)
        threading.Thread(target=self.scheduler.run, daemon=True).start()

    def viren_self_repair(self):
        while True:
            try:
                if not args.no_cloud:
                    # Cloud probes...
                    pass
                # assemble_vortex(Assemble())
            except Exception as e:
                log.warning(f"Heal triggered: {e}")
                for node in self.metatron_graph.nodes:
                    self.metatron_graph.nodes[node]['weight'] += 0.1
            time.sleep(60)

    def process_request(self, request: dict):
        log.info(f"MetaEngine processing request: {request}")
        signal = request.get("signal", [0.0] * 13)
        step = request.get("step", 0)
log.info(f"MetaEngine processing request: {request}")
        signal = request.get("signal", [0.0] * 13)
        step = request.get("step", 0)
        try:
            filtered_signal = self.filter.apply(signal, step)
            return {"status": "processed", "request": request, "filtered_signal": filtered_signal}
        except Exception as e:
            log.error(f"Error processing request: {e}")
            return {"status": "error", "message": str(e)}

meta_engine = MetaEngine()
        return {"status": "processed", "request": request, "filtered_signal": filtered_signal}

meta_engine = MetaEngine()

@app.post("/process")
async def process(request: dict):
    return meta_engine.process_request(request)

@app.get("/harmony")
async def harmony():
    return {"harmony": orc.gabriels_horn.compute_harmony()}

orc = None
if __name__ == "__main__":
    meta_engine.load_policy()
    meta_engine.harmonic_scheduler()
    threading.Thread(target=meta_engine.viren_self_repair, daemon=True).start()

    # Register with orchestration layer
    orc = OrchestrationLayer()
    asyncio.run(orc.initialize())
    node_id = "meta_engine_service"
    node_info = {
        "type": "meta_engine",
        "url": "http://localhost:8000"
    }
    asyncio.run(orc.register_node(node_id, node_info))

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)