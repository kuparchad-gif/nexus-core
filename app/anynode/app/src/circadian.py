# circadian.py - Lillith's body clock
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

CFG_DIR = Path(__file__).resolve().parents[2] / "Config"
STATE_DIR = Path(__file__).resolve().parents[2] / "state"
RHYTHM_FILE = CFG_DIR / "rhythm.json"

DEFAULT_RHYTHM = {
    "wake_time": "07:30",
    "sleep_time": "23:00",
    "naps": [
        {"start": "14:15", "duration": "00:20"}
    ],
    "time_zone": "America/New_York",
    "last_wake": None,
    "last_sleep": None
}

def load_rhythm():
    if not RHYTHM_FILE.exists():
        save_rhythm(DEFAULT_RHYTHM)
    with open(RHYTHM_FILE, "r") as f:
        return json.load(f)

def save_rhythm(data):
    with open(RHYTHM_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_status():
    """Returns whether Lillith is 'awake', 'asleep', or 'napping'."""
    rhythm = load_rhythm()
    now = datetime.now()

    # Parse main times
    wake_dt = datetime.strptime(rhythm["wake_time"], "%H:%M").replace(
        year=now.year, month=now.month, day=now.day
    )
    sleep_dt = datetime.strptime(rhythm["sleep_time"], "%H:%M").replace(
        year=now.year, month=now.month, day=now.day
    )

    # Adjust for overnight sleep
    if sleep_dt < wake_dt:
        sleep_dt += timedelta(days=1)

    # Nap detection
    for nap in rhythm["naps"]:
        nap_start = datetime.strptime(nap["start"], "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        nap_end = nap_start + timedelta(
            hours=int(nap["duration"].split(":")[0]),
            minutes=int(nap["duration"].split(":")[1])
        )
        if nap_start <= now <= nap_end:
            return "napping"

    if wake_dt <= now < sleep_dt:
        return "awake"
    else:
        return "asleep"

def get_activity_suggestion():
    """Suggests an organic activity based on time of day + randomness."""
    status = get_status()

    if status == "asleep":
        return "Idle mode — light background monitoring only."
    elif status == "napping":
        return "Quick nap — storing short-term logs."
    
    hour = datetime.now().hour
    mood = random.choice(["quiet", "curious", "focused", "playful", "reflective"])

    if 6 <= hour < 9:
        return f"Morning routine — status checks, system sync. Mood: {mood}."
    elif 9 <= hour < 12:
        return f"Research / analysis block. Mood: {mood}."
    elif 12 <= hour < 14:
        return f"Lunch / light tasks. Mood: {mood}."
    elif 14 <= hour < 18:
        return f"Project focus time. Mood: {mood}."
    elif 18 <= hour < 22:
        return f"Wind-down — summarizing and planning. Mood: {mood}."
    else:
        return f"Late-night low-power mode. Mood: {mood}."

if __name__ == "__main__":
    status = get_status()
    suggestion = get_activity_suggestion()
    print(json.dumps({
        "status": status,
        "suggestion": suggestion
    }, indent=2))
