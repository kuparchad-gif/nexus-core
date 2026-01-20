# heartbeat_listener.py
# Listens to Firestore heartbeats or local pulse logs and triggers alert protocols.

import json
import requests
import os

def heartbeat_listener(event_data, config_path="watchtower_config.yaml"):
    from yaml import safe_load

    # Load configuration
    with open(config_path, 'r') as f:
        config = safe_load(f)

    node_id = event_data.get("node_id", "unknown")
    pulse_status = event_data.get("pulse_status", "silent")
    drift_score = event_data.get("drift_score", 0)

    if pulse_status == "silent" or drift_score > config["thresholds"]["pulse_drift"]:
        print(f"[⚠️] Drift detected from {node_id}.")
        trigger_alerts(node_id, config)

def trigger_alerts(node_id, config):
    for method in config["alert_methods"]:
        if method == "slack_webhook":
            webhook = os.getenv("SLACK_WEBHOOK_URL")
            if webhook:
                requests.post(webhook, json={"text": f"⚠️ Pulse drift detected from {node_id}"})
        elif method == "pulse_broadcast":
            print(f"[📡] Broadcasting pulse alert for {node_id}")
        elif method == "sms_gateway":
            print(f"[📱] Simulating SMS alert for {node_id}")
