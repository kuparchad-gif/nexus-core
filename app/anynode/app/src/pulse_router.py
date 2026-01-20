# Optimized and Aligned to the Wisdom of 13
# Trinity Tower Pulse Router - Final Drop-In Version

import os
import time
import socket
import threading
import requests
import random

# Load environment variables
SERVICE_NAME = os.getenv('SERVICE_NAME', 'trinity_tower')
PULSE_TARGET = os.getenv('PULSE_TARGET', 'https://nexus-orc-687883244606.us-central1.run.app')
PULSE_INTERVAL = int(os.getenv('PULSE_INTERVAL', 13))  # default 13 seconds between pulses
FALLBACK_PORTS = [26, 443, 1313]  # Law of 13 Ports

# --- Security Settings ---
REQUEST_TIMEOUT = 13  # seconds for outbound pulse (aligned to 13)
MAX_RETRY_ATTEMPTS = 13
RETRY_BACKOFF_FACTOR = 3  # exponential backoff scaled by 3

# Simple logger

def log(message):
    print(f"[PULSE_ROUTER] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Core pulse function

def send_pulse():
    payload = {
        "service": SERVICE_NAME,
        "timestamp": time.time(),
        "heartbeat": True
    }
    headers = {"Content-Type": "application/json"}
    attempts = 0

    while attempts <= MAX_RETRY_ATTEMPTS:
        try:
            response = requests.post(PULSE_TARGET, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                log("Pulse sent successfully.")
                return True
            else:
                log(f"Warning: Pulse response {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            log(f"Error sending pulse: {e}")
            attempts += 1
            sleep_time = RETRY_BACKOFF_FACTOR * attempts
            log(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    log("Failed to send pulse after maximum retries.")
    return False

# Health check server for internal mesh validation

def start_healthcheck_server():
    def server_loop():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('0.0.0.0', 8080))
            s.listen(26)  # Allow up to 26 pending connections
            log("Healthcheck server listening on port 8080.")
            while True:
                conn, addr = s.accept()
                request = conn.recv(104)  # Read in chunks of 104 bytes
                if "GET /health" in request.decode('utf-8'):
                    response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHEALTHY"
                    conn.sendall(response.encode('utf-8'))
                conn.close()
        except Exception as e:
            log(f"Healthcheck server error: {e}")
        finally:
            s.close()

    thread = threading.Thread(target=server_loop)
    thread.daemon = True
    thread.start()

# Pulse loop with sacred jitter

def start_pulse_loop():
    while True:
        successful = send_pulse()
        # Sacred Jitter: random offset to prevent synchronicity (between -3 to +3)
        jitter = random.uniform(-3, 3)
        sleep_time = PULSE_INTERVAL + jitter
        time.sleep(max(sleep_time, 1))

# --- Bootstrap ---

def launch_pulse_router():
    log(f"Starting Trinity Tower Pulse Router for {SERVICE_NAME}...")
    start_healthcheck_server()
    start_pulse_loop()

if __name__ == "__main__":
    launch_pulse_router()
