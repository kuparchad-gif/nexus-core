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

# ... (Rest: auth_check, enforce_policy, probe_health with no-cloud flag skips, middleware, endpoints as prior but with more try/except)

# Self-Repair Enhanced
def viren_self_repair():
    while True:
        try:
            if not args.no_cloud:
                # Cloud probes...
                pass
            assemble_vortex(Assemble())
        except Exception as e:
            log.warning(f"Heal triggered: {e}")
            for node in metatron_graph.nodes:
                metatron_graph.nodes[node]['weight'] += 0.1  # Boost
        time.sleep(60)

if __name__ == "__main__":
    load_policy()
    harmonic_scheduler()
    threading.Thread(target=viren_self_repair, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)