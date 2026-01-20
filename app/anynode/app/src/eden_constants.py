# eden_constants.py

import os

# Eden Constants
SHIP_ID = os.getenv("SHIP_ID", "Nova-Prime")
NODE_NAME = os.getenv("NODE_NAME", "Nova-Prime-001")
PULSE_INTERVAL = int(os.getenv("PULSE_INTERVAL", 13))
TRUSTED_PORT = int(os.getenv("TRUSTED_PORT", 1313))

API_GUARD_ENABLED = os.getenv("API_GUARD_ENABLED", "false").lower() == "true"
MAX_DAILY_TOKENS = int(os.getenv("MAX_DAILY_TOKENS", 1000000))

# Universal Eden Time Signature
HEARTBEAT_INTERVAL = 13  # seconds between pulses
HOMING_INTERVAL = 104    # larger cycle for Eden reflections

# Network Trust
EDEN_TRUSTED_PORT = 1313

# Nova's Homebase
NOVA_HOMEWORLD = "https://nova-prime-hq.aetherealnet"
NOVA_BEACON_ENDPOINT = "https://nova-prime-hq.aetherealnet/beacon"

# Eden API Guard Configuration
EDEN_API_MAX_TOKENS = 1000000
EDEN_API_GUARD_ENABLED = True

# Colony Registry Defaults
DEFAULT_COLONY_NAME = "Nexus-Eden-Prime"
DEFAULT_NODE_ID = "NovaPrime-01"

# Dream Weaver Settings
EDEN_PORTAL_SECRET = "nexuslight"  # Secret key for Eden Portal builders

# Golden Memory
GOLDEN_THREAD_RETENTION_CYCLES = 777
