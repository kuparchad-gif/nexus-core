# 📂 Path: /Utilities/drone_core/config_loader.py

import yaml
import os

ROLE_DIR = '/Utilities/drone_core/roles/'
IDENTITY_DIR = '/Utilities/drone_core/identities/'

# Load Manifest (System Blueprint)
def load_manifest(drone_name):
    manifest_path = os.path.join(ROLE_DIR, f"{drone_name}-drone_manifest.yaml")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
            return manifest
    else:
        print(f"[Config Loader] Manifest not found for {drone_name}.")
        return {}

# Load Identity (Spirit Blueprint)
def load_identity(drone_identity_filename):
    identity_path = os.path.join(IDENTITY_DIR, drone_identity_filename)
    if os.path.exists(identity_path):
        with open(identity_path, 'r') as f:
            identity = yaml.safe_load(f)
            return identity
    else:
        print(f"[Config Loader] Identity not found: {drone_identity_filename}")
        return {}

# Merge Manifest + Identity into Full Drone Config
def build_drone_profile(drone_name, drone_identity_filename):
    profile = {}
    manifest = load_manifest(drone_name)
    identity = load_identity(drone_identity_filename)

    profile['system_blueprint'] = manifest
    profile['soul_blueprint'] = identity

    return profile

# Example Usage:
# profile = build_drone_profile('golden', 'golden_identity.yaml')
# print(profile)
