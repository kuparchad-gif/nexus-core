# guardian_test.py
# Simulates a pulse drift or missing heartbeat event

from heartbeat_listener import heartbeat_listener

test_event = {
    "node_id": "colony-47",
    "pulse_status": "silent",
    "drift_score": 9.3
}

heartbeat_listener(test_event)