# 📂 Path: /Systems/engine/memory_service/blueprint_builder.py

import json
import os

BLUEPRINT_PATH  =  '/memory/blueprints/memory_blueprint.json'
os.makedirs('/memory/blueprints/', exist_ok = True)

def build_blueprint(assignments):
    if os.path.exists(BLUEPRINT_PATH):
        with open(BLUEPRINT_PATH, 'r') as f:
            blueprint  =  json.load(f)
    else:
        blueprint  =  {}

    blueprint.update(assignments)

    with open(BLUEPRINT_PATH, 'w') as f:
        json.dump(blueprint, f, indent = 2)

    print(f"[Blueprint Builder] Updated blueprint with {len(assignments)} new shards.")

# Example Usage:
# build_blueprint(dispatch_shards(list_of_shard_files))
