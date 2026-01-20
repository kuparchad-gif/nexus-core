# bookworms/system_worm.py
import platform, socket, json, time
try:
    import psutil
except Exception:
    psutil = None

def collect_system_snapshot():
    hostname = socket.gethostname()
    os_str = f"{platform.system()} {platform.release()}"
    kernel = platform.version()
    cpu = platform.processor() or platform.machine()
    cores = psutil.cpu_count(logical=False) if psutil else None
    threads = psutil.cpu_count(logical=True) if psutil else None
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None
    gpus = "AMD/NVIDIA/None (enumerate via vendor CLI in production)"
    return {
        "id": hostname,
        "hostname": hostname,
        "os": os_str,
        "kernel": kernel,
        "cpu_model": cpu,
        "cpu_cores": cores,
        "cpu_threads": threads,
        "total_ram_gb": ram_gb,
        "gpus": gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
