# C:\Projects\LillithNew\src\utils\check_lillith.py
import os
import json
import subprocess
import requests
from datetime import datetime

class LillithChecker:
    def __init__(self):
        self.root = "C:\\Projects\\LillithNew"
        self.logs_dir = os.path.join(self.root, "logs")
        self.pid_file = os.path.join(self.root, "state", "lillith_pids.json")
        self.loki_endpoint = "http://loki:3100/loki/api/v1/push"

    def log_to_loki(self, message):
        try:
            payload = {
                "streams": [{"stream": {"job": "lillith_checker"}, "values": [[str(int(time.time() * 1e9)), message]]}]
            }
            requests.post(self.loki_endpoint, json=payload)
        except Exception as e:
            print(f"Loki logging failed: {str(e)}")

    def check_tasks(self):
        tasks = ["Lillith-Launch-OnLogon", "Lillith-Watchdog-Always", "Lillith-Update-Daily"]
        for task in tasks:
            try:
                result = subprocess.run(["schtasks", "/Query", "/TN", task], capture_output=True, text=True, check=True)
                self.log_to_loki(f"Task {task} is active.")
            except subprocess.CalledProcessError:
                self.log_to_loki(f"Task {task} not found!")

    def check_logs(self):
        if not os.path.exists(self.logs_dir):
            self.log_to_loki("Logs directory missing!")
            return
        log_files = [f for f in os.listdir(self.logs_dir) if f.endswith(".log")]
        if not log_files:
            self.log_to_loki("No log files found!")
        else:
            self.log_to_loki(f"Found {len(log_files)} log files in {self.logs_dir}")

    def check_pids(self):
        if not os.path.exists(self.pid_file):
            self.log_to_loki("PID file missing!")
            return
        with open(self.pid_file, "r") as f:
            pids = json.load(f)
        for service, pid in pids.items():
            try:
                result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=True)
                if str(pid) in result.stdout:
                    self.log_to_loki(f"Service {service} (PID {pid}) is running.")
                else:
                    self.log_to_loki(f"Service {service} (PID {pid}) not running!")
            except subprocess.CalledProcessError:
                self.log_to_loki(f"Service {service} (PID {pid}) not running!")

    def run(self):
        self.log_to_loki("Starting Lillith sanity check...")
        self.check_tasks()
        self.check_logs()
        self.check_pids()
        self.log_to_loki("Sanity check complete.")

if __name__ == "__main__":
    checker = LillithChecker()
    checker.run()