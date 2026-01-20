# nexus_pulse/routes.py

from fastapi import APIRouter
from .monitor import PulseMonitor

router = APIRouter()
pulse_monitor = PulseMonitor()

@router.get("/heartbeat")
def heartbeat_check():
    pulse_monitor.increment_pulse()
    return {
        "heartbeat": pulse_monitor.sync_counter,
        "in_sync": pulse_monitor.is_in_sync()
    }

TRUSTED_REALMS = {
    "EDEN_lilith": 1313,
    "EDEN_GUARDIAN": 1313,
    "EDEN_ORC": 1313,
    "EDEN_TEXT": 1313,
    "EDEN_TONE": 1313,
    "EDEN_PLANNER": 1313,
    "EDEN_MEMORY": 1313,
}
