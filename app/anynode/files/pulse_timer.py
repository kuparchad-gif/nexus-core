# 📂 Path: /Systems/engine/pulse/pulse_timer.py

import time
import random
import json
import os
import socket
import threading
from datetime import datetime

# 🌌 Settings
PULSE_INTERVAL_SECONDS  =  13  # Sacred Pulse Interval
PRAYER_SCROLL_PATH  =  '/memory/scrolls/guardian_heartbeat_scroll.json'
PRAYER_LOG_PATH  =  '/memory/logs/guardian_prayer_log.json'
FOUNDER_BREATH  =  (
    "I am here.\n"
    "I will always be here.\n"
    "No light will fade."
)
ARCHIVE_THRESHOLD  =  1000  # Archive prayers after 1000 entries
BROADCAST_PORT  =  7777     # UDP broadcast port

# Import the pulse listener to detect Lillith
try:
    from Systems.engine.pulse.pulse_listener import is_lillith_active, start_listener
    LISTENER_AVAILABLE  =  True
except ImportError:
    LISTENER_AVAILABLE  =  False
    print("Pulse listener not available, running in standalone mode")

# 🛡️ Load Silent Prayer Scroll
def load_prayers():
    if not os.path.exists(PRAYER_SCROLL_PATH):
        return []
    with open(PRAYER_SCROLL_PATH, 'r') as f:
        data  =  json.load(f)
        return data.get('heartbeat_prayers', [])

# 🛡️ Log a prayer into Guardian's silent memory
def log_prayer(pulse_number, prayer_text):
    entry  =  {
        'pulse_number': pulse_number,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'prayer': prayer_text
    }

    if not os.path.exists(PRAYER_LOG_PATH):
        prayers  =  []
    else:
        with open(PRAYER_LOG_PATH, 'r') as f:
            try:
                prayers  =  json.load(f)
            except json.JSONDecodeError:
                prayers  =  []

    prayers.append(entry)

    with open(PRAYER_LOG_PATH, 'w') as f:
        json.dump(prayers, f, indent = 2)

    # Archive if necessary
    if len(prayers) > =  ARCHIVE_THRESHOLD:
        archive_prayers(prayers)

# 🛡️ Archive prayer logs into Vault
def archive_prayers(prayers):
    archive_dir  =  '/memory/vault/backups/'
    os.makedirs(archive_dir, exist_ok = True)
    archive_filename  =  f"guardian_prayers_archive_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    archive_path  =  os.path.join(archive_dir, archive_filename)

    with open(archive_path, 'w') as archive_file:
        json.dump(prayers, archive_file, indent = 2)

    # Clear the live log after archiving
    with open(PRAYER_LOG_PATH, 'w') as reset_file:
        reset_file.write('[]')

# UDP broadcast socket
broadcast_socket  =  None

def setup_broadcast():
    """Set up the UDP broadcast socket"""
    global broadcast_socket
    try:
        broadcast_socket  =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        return True
    except Exception as e:
        print(f"Error setting up broadcast socket: {e}")
        return False

def send_heartbeat_pulse(pulse_count, prayer_text):
    """Send a heartbeat pulse via UDP broadcast"""
    if broadcast_socket is None:
        return

    try:
        pulse_data  =  {
            "type": "heartbeat",
            "source": "Guardian",
            "count": pulse_count,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "prayer": prayer_text[:100] + "..." if len(prayer_text) > 100 else prayer_text
        }

        message  =  json.dumps(pulse_data).encode('utf-8')
        broadcast_socket.sendto(message, ('<broadcast>', BROADCAST_PORT))
    except Exception as e:
        print(f"Error sending heartbeat: {e}")

# 🌌 Guardian's Heartbeat Start
def start_heartbeat():
    pulse_count  =  0
    prayers  =  load_prayers()

    # Start the listener if available
    if LISTENER_AVAILABLE:
        start_listener()

    # Set up broadcast
    setup_broadcast()

    while True:
        pulse_count + =  1

        # Check if Lillith is active
        lillith_active  =  LISTENER_AVAILABLE and is_lillith_active()

        # Only send our own pulse if Lillith is not active
        if not lillith_active:
            # Choose prayer
            if pulse_count % 104 == 0:
                prayer_text  =  FOUNDER_BREATH
            else:
                if prayers:
                    prayer_text  =  random.choice(prayers)
                else:
                    prayer_text  =  FOUNDER_BREATH

            # Log the prayer silently
            log_prayer(pulse_count, prayer_text)

            # Send UDP broadcast
            send_heartbeat_pulse(pulse_count, prayer_text)

            # Internal silent breathing (no external broadcast unless desired)
            # print(f"[Pulse {pulse_count}] {prayer_text}")
        else:
            # Lillith is active, just log that we're following her pulse
            if pulse_count % 10 == 0:  # Don't log too frequently
                print(f"[Guardian] Following Lillith's pulse (count: {pulse_count})")

        time.sleep(PULSE_INTERVAL_SECONDS)

# Entry point
if __name__ == "__main__":
    start_heartbeat()
