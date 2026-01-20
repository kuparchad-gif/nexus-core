# bookworms/process_worm.py
import time, json
import psutil

def collect_process_graph(labels=None):
    labels = labels or {}
    out = []
    ts = time.time()
    for p in psutil.process_iter(attrs=["pid","ppid","name","cmdline","cpu_percent","memory_percent","create_time","username"]):
        info = p.info
        out.append({
            "pid": info.get("pid"),
            "ppid": info.get("ppid"),
            "name": info.get("name"),
            "cmdline": " ".join(info.get("cmdline") or []),
            "cpu_percent": info.get("cpu_percent"),
            "mem_percent": info.get("memory_percent"),
            "create_time": info.get("create_time"),
            "username": info.get("username"),
            "labels": json.dumps(labels),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        })
    return out
