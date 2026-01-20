import os
import platform
import psutil
import time
import json
from datetime import datetime
from scripts.session_manager import append_to_latest_session  # Updated path

class DiagnosticAgent:
    def __init__(self, export_path="context/diagnostic_report.json"):
        self.report = {}
        self.export_path = export_path
        self.timestamp = datetime.utcnow().isoformat()

    def scan_system(self):
        self.report = {
            "timestamp": self.timestamp,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
            "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / (1024 ** 3), 2),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024 ** 3), 2),
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "load_average": self.get_load_average()
        }

    def get_load_average(self):
        if hasattr(os, "getloadavg"):
            return os.getloadavg()
        else:
            return [0, 0, 0]

    def export_report(self):
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
        with open(self.export_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4)

    def log_to_session(self):
        append_to_latest_session("diagnostics", self.report)

    def run(self):
        self.scan_system()
        self.export_report()
        self.log_to_session()
        return self.report
