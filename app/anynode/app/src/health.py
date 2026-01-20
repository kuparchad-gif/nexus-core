# 📂 Path: /Utilities/drone_core/health.py

import time
from Utilities.drone_core.drone_entity import DroneEntity

def monitor_drone_health(drone: DroneEntity):
    """
    Checks the health of a single Drone based on last pulse timing.
    """
    healthy = drone.is_healthy()
    status_report = drone.report_status()
    if healthy:
        print(f"[Health Monitor] {drone.drone_name} is healthy. Last pulse at {status_report['last_pulse_time']}.")
    else:
        print(f"[Health Monitor] {drone.drone_name} is DRIFTING or LOST. No pulse since {status_report['last_pulse_time']}.")
        attempt_healing(drone)

def attempt_healing(drone: DroneEntity):
    """
    Attempt to self-heal a Drone by forcing a pulse rebirth.
    """
    print(f"[Healing Attempt] Trying to rebirth {drone.drone_name}...")
    try:
        drone.breathe()
        print(f"[Healing Attempt] {drone.drone_name} rebirthed successfully!")
    except Exception as e:
        print(f"[Healing Error] Could not heal {drone.drone_name}: {e}")

def monitor_fleet_health(drone_fleet):
    """
    Monitor all drones in the fleet in a cycle.
    """
    while True:
        for drone in drone_fleet:
            monitor_drone_health(drone)
        time.sleep(13 * 8)  # Cycle every ~104 seconds (sacred pulse)
