
#!/usr/bin/env python3
# Metatron Watchdog MT: multithreaded, constant scanning, Prometheus metrics
#
# - Launches one watcher thread per service (process/windows_service/scheduled_task/systemd)
# - Continuous health probes (per-service interval)
# - Crash counters with escalation: alert core, wake Viren, push to Loki
# - Sniffer + Heal restart modes after threshold
# - /metrics endpoint (Prometheus/OpenMetrics text)
# - Optional "ghost hunter" thread to discover stray tasks/services
#
# Usage:
#   python watchdog_mt.py --manifest watchdog_mt.manifest.json
#   python watchdog_mt.py --manifest watchdog_mt.manifest.json --ghosts
#
import argparse, json, os, platform, subprocess, sys, time, shlex, re, threading, socket
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

IS_WIN = platform.system().lower().startswith("win")
IS_LINUX = platform.system().lower() == "linux"
IS_MAC = platform.system().lower() == "darwin"

# -------------------------- Utils --------------------------
def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json(path, obj):
    try:
        ensure_parent(path)
        tmp = path + ".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(obj,f,indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[WARN] failed to write {path}: {e}")

# -------------------------- Metrics --------------------------
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = {} # name -> {(label_tuple): value}
        self.gauges   = {}
        self.last     = {} # last observed values (latencies, etc.)

    def _key(self, labels):
        # labels: dict -> tuple sorted items
        return tuple(sorted((labels or {}).items()))

    def inc(self, name, labels=None, by=1):
        key = self._key(labels)
        with self.lock:
            self.counters.setdefault(name, {})
            self.counters[name][key] = self.counters[name].get(key, 0) + by

    def set(self, name, value, labels=None):
        key = self._key(labels)
        with self.lock:
            self.gauges.setdefault(name, {})
            self.gauges[name][key] = float(value)

    def observe_last(self, name, value, labels=None):
        key = self._key(labels)
        with self.lock:
            self.last.setdefault(name, {})
            self.last[name][key] = float(value)

    def render_prom(self):
        # very simple OpenMetrics text
        with self.lock:
            lines = []
            for n, m in self.counters.items():
                lines.append(f"# TYPE {n} counter")
                for labels_t, val in m.items():
                    label_text = ",".join([f'{k}="{v}"' for k,v in labels_t])
                    if label_text:
                        lines.append(f'{n}{{{label_text}}} {val}')
                    else:
                        lines.append(f'{n} {val}')
            for n, m in self.gauges.items():
                lines.append(f"# TYPE {n} gauge")
                for labels_t, val in m.items():
                    label_text = ",".join([f'{k}="{v}"' for k,v in labels_t])
                    if label_text:
                        lines.append(f'{n}{{{label_text}}} {val}')
                    else:
                        lines.append(f'{n} {val}')
            for n, m in self.last.items():
                lines.append(f"# TYPE {n} gauge")
                for labels_t, val in m.items():
                    label_text = ",".join([f'{k}="{v}"' for k,v in labels_t])
                    if label_text:
                        lines.append(f'{n}{{{label_text}}} {val}')
                    else:
                        lines.append(f'{n} {val}')
            return "\n".join(lines) + "\n"

METRICS = Metrics()

# metric names
C_CHECKS    = "wd_checks_total"
C_RESTARTS  = "wd_restarts_total"
C_ESCALATE  = "wd_escalations_total"
C_HEALTHERR = "wd_health_failures_total"
G_UP        = "wd_service_up"
G_CRASHES   = "wd_crash_streak"
L_HEALTH_S  = "wd_health_latency_seconds"
L_RESTART_S = "wd_restart_duration_seconds"

# -------------------------- HTTP helpers --------------------------
def http_post_json(url, obj, timeout=5):
    try:
        data = json.dumps(obj).encode("utf-8")
        req = Request(url, data=data, headers={"content-type":"application/json"})
        t0 = time.perf_counter()
        with urlopen(req, timeout=timeout) as r:
            METRICS.observe_last(L_HEALTH_S, time.perf_counter()-t0, {"url":url,"op":"post"})
            return True, getattr(r, "status", 200)
    except Exception as e:
        return False, str(e)


# -------------------------- CPU settings --------------------------
def apply_cpu_settings(pid, cpu_spec):
    if not cpu_spec:
        return
    try:
        import psutil
    except Exception:
        # psutil not installed; skip
        return
    try:
        p = psutil.Process(pid)
        # Affinity
        aff = cpu_spec.get("affinity")
        if isinstance(aff, list) and len(aff) > 0:
            try:
                p.cpu_affinity(aff)
            except Exception:
                pass
        # Priority / nice
        pr = str(cpu_spec.get("priority","")).lower()
        try:
            if IS_WIN:
                import psutil as _ps
                mapping = {
                    "idle": _ps.IDLE_PRIORITY_CLASS,
                    "below_normal": _ps.BELOW_NORMAL_PRIORITY_CLASS,
                    "normal": _ps.NORMAL_PRIORITY_CLASS,
                    "above_normal": _ps.ABOVE_NORMAL_PRIORITY_CLASS,
                    "high": _ps.HIGH_PRIORITY_CLASS,
                    "realtime": _ps.REALTIME_PRIORITY_CLASS
                }
                if pr in mapping:
                    p.nice(mapping[pr])
            else:
                # On Unix, lower nice value = higher priority (-20 .. 19)
                mapping = {"idle":19, "below_normal":10, "normal":0, "above_normal":-5, "high":-10, "realtime":-15}
                if pr in mapping:
                    p.nice(mapping[pr])
        except Exception:
            pass
        # I/O nice (Linux)
        ion = str(cpu_spec.get("ionice","")).lower()
        if ion and not IS_WIN:
            classes = {"idle":3, "best_effort":2, "realtime":1}
            if ion in classes:
                try:
                    p.ionice(classes[ion])
                except Exception:
                    pass
    except Exception:
        pass
def loki_push(url: str, labels: dict, line: str):
    try:
        ns = str(int(time.time() * 1_000_000_000))
        payload = {"streams":[{"stream": labels, "values":[[ns, line]]}]}
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"content-type":"application/json"})
        with urlopen(req, timeout=3) as r:
            return r.status in (200, 204)
    except Exception:
        return False

# -------------------------- Health --------------------------
def check_http_health(spec, labels):
    if not spec: 
        METRICS.set(G_UP, 1, labels)
        return True, "no health spec"
    url = spec.get("http"); timeout = int(spec.get("timeout", 5))
    expect_field = spec.get("expect_json_field"); expect_value = spec.get("expect_value")
    if not url:
        METRICS.set(G_UP, 1, labels)
        return True, "no health url"
    try:
        req = Request(url, headers={"user-agent":"watchdog-mt/1.0"})
        t0 = time.perf_counter()
        with urlopen(req, timeout=timeout) as r:
            body = r.read()
            METRICS.observe_last(L_HEALTH_S, time.perf_counter()-t0, {"service":labels["service"]})
            if expect_field:
                try:
                    obj = json.loads(body.decode("utf-8","ignore"))
                    ok = (obj.get(expect_field) == expect_value) if expect_value is not None else (expect_field in obj)
                    METRICS.set(G_UP, 1 if ok else 0, labels)
                    if not ok: METRICS.inc(C_HEALTHERR, labels)
                    return (ok, f"field:{expect_field}=={expect_value} -> {ok}")
                except Exception as e:
                    METRICS.set(G_UP, 0, labels); METRICS.inc(C_HEALTHERR, labels)
                    return False, f"json parse fail: {e}"
            METRICS.set(G_UP, 1, labels)
            return True, f"http {getattr(r,'status',200)}"
    except Exception as e:
        METRICS.set(G_UP, 0, labels); METRICS.inc(C_HEALTHERR, labels)
        return False, f"health error: {e}"

# -------------------------- Process Discovery --------------------------
def list_processes():
    try:
        import psutil  # type: ignore
        for p in psutil.process_iter(attrs=["pid","name","cmdline"]):
            name = p.info.get("name") or ""
            cmd = " ".join(p.info.get("cmdline") or [])
            yield (p.pid, name, cmd)
    except Exception:
        if IS_WIN:
            cp = subprocess.run(["wmic","process","get","ProcessId,Name,CommandLine","/format:csv"],
                                capture_output=True, text=True, errors="ignore")
            for line in cp.stdout.splitlines():
                parts = [x for x in line.split(",")]
                if len(parts) < 4: continue
                name = parts[-3].strip(); cmd  = parts[-2].strip(); pid  = parts[-1].strip()
                try: pid = int(pid)
                except: pid = None
                if name or cmd:
                    yield (pid, name, cmd)
        else:
            cp = subprocess.run(["ps","-eo","pid,comm,args"], capture_output=True, text=True, errors="ignore")
            for line in cp.stdout.splitlines()[1:]:
                try:
                    pid_str, rest = line.strip().split(" ", 1)
                    pid = int(pid_str)
                    if " " in rest: name, cmd = rest.split(" ", 1)
                    else: name, cmd = rest, rest
                    yield (pid, name, cmd)
                except Exception:
                    continue

def is_process_running(match):
    name = (match or {}).get("process_name","").lower()
    frag = (match or {}).get("cmdline_contains","").lower()
    for pid, pname, cmd in list_processes():
        if name and (pname or "").lower() != name: continue
        if frag and frag not in (cmd or "").lower(): continue
        return True
    return False

def start_command(start_spec, extra_env=None, extra_args=None):
    if not start_spec: return False, "no start spec"
    cmd = start_spec.get("command")
    if not cmd: return False, "no start command"
    cwd = start_spec.get("cwd") or None
    env = os.environ.copy()
    env.update(start_spec.get("env") or {})
    if extra_env: env.update(extra_env)
    if extra_args: cmd = f"{cmd} {extra_args}"
    try:
        t0 = time.perf_counter()
        if IS_WIN:
            DETACHED = 0x00000008
            subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=DETACHED, shell=True)
        else:
            subprocess.Popen(shlex.split(cmd), cwd=cwd, env=env, start_new_session=True)
        METRICS.observe_last(L_RESTART_S, time.perf_counter()-t0, {})
        return True, "spawned"
    except Exception as e:
        return False, f"spawn error: {e}"

# Windows services / tasks / systemd
def win_service_status(name):
    cp = subprocess.run(["sc","query",name], capture_output=True, text=True, errors="ignore")
    if cp.returncode != 0: return "missing"
    m = re.search(r"STATE\s*:\s*\d+\s+(\w+)", cp.stdout)
    return m.group(1).lower() if m else "unknown"
def win_service_start(name):
    cp = subprocess.run(["sc","start",name], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())
def win_service_enable(name):
    cp = subprocess.run(["sc","config",name,"start=","auto"], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())
def schtask_info(name):
    cp = subprocess.run(["schtasks","/Query","/TN",name,"/V","/FO","LIST"], capture_output=True, text=True, errors="ignore")
    if cp.returncode != 0: return {"exists": False}
    info = {"exists": True, "enabled": None, "status": None, "next_run_time": None}
    for line in cp.stdout.splitlines():
        if line.startswith("Enabled:"): info["enabled"] = line.split(":",1)[1].strip()
        elif line.startswith("Status:"): info["status"] = line.split(":",1)[1].strip()
        elif line.startswith("Next Run Time:"): info["next_run_time"] = line.split(":",1)[1].strip()
    return info
def schtask_enable(name):
    cp = subprocess.run(["schtasks","/Change","/TN",name,"/ENABLE"], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())
def schtask_run(name):
    cp = subprocess.run(["schtasks","/Run","/TN",name], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())
def systemd_status(unit):
    cp = subprocess.run(["systemctl","is-active",unit], capture_output=True, text=True, errors="ignore")
    if cp.returncode != 0: return "missing"
    return cp.stdout.strip()
def systemd_start(unit):
    cp = subprocess.run(["systemctl","start",unit], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())
def systemd_enable(unit):
    cp = subprocess.run(["systemctl","enable",unit], capture_output=True, text=True, errors="ignore")
    return (cp.returncode==0, cp.stdout.strip() or cp.stderr.strip())

# -------------------------- Escalation --------------------------
def escalate(service_name, reason, cfg, host):
    payload = {"ts": now_ts(), "service": service_name, "host": host, "reason": reason}
    for url in cfg.get("alert_urls", []):
        ok, status = http_post_json(url, payload)
        print(f"[ESC] alert -> {url} : {ok} ({status})")
    for url in cfg.get("wake_urls", []):
        ok, status = http_post_json(url, {"wake": service_name, **payload})
        if not ok: ok, status = http_post_json(url, {"wake": service_name})
        print(f"[ESC] wake -> {url} : {ok} ({status})")
    loki = cfg.get("loki")
    if loki and loki.get("url"):
        labels = {"app":"watchdog", "host":host, "service":service_name, **(loki.get("labels") or {})}
        ok = loki_push(loki["url"], labels, f"ESCALATE {service_name} {reason}")
        print(f"[ESC] loki -> {ok}")

# -------------------------- Watcher --------------------------
def watch_service(svc, state, host):
    name = svc.get("name","<unnamed>")
    typ  = (svc.get("type") or "process").lower()
    autostart = bool(svc.get("autostart", True))
    grace = int(svc.get("grace_seconds", 8))
    interval = int(svc.get("interval", 5))  # default 5s probes
    health = svc.get("health")
    start_spec = svc.get("start") or {}
    labels = {"service": name}

    policy = svc.get("crash_policy") or {}
    threshold = int(policy.get("threshold", 3))
    backoff = policy.get("backoff_sec", [5, 15, 30])
    escalate_cfg = policy.get("escalate") or {}

    sniffer = svc.get("sniffer") or {}
    heal = svc.get("heal") or {}

    crashes = state.setdefault("crashes", {}).get(name, 0)

    while True:
        try:
            METRICS.inc(C_CHECKS, labels)
            ok = False

            if typ == "process":
                running = is_process_running(svc.get("match") or {})
                if running:
                    ok, hmsg = check_http_health(health, labels)
                    if ok:
                        METRICS.set(G_CRASHES, 0, labels); state["crashes"][name] = 0
                    else:
                        METRICS.inc(C_HEALTHERR, labels)
                else:
                    if not autostart:
                        METRICS.set(G_UP, 0, labels)
                        time.sleep(interval); continue

                    # choose mode based on crashes
                    extra_env=None; extra_args=None
                    crashes = state["crashes"].get(name, 0)
                    if crashes >= threshold:
                        extra_env = sniffer.get("env"); extra_args = sniffer.get("extra_args")
                    ok, msg, _ = start_command(start_spec, extra_env=extra_env, extra_args=extra_args, cpu_spec=svc.get('cpu'))
                    METRICS.inc(C_RESTARTS, {"service":name,"mode":"normal" if crashes<threshold else "sniffer"})
                    time.sleep(grace)
                    ok, hmsg = check_http_health(health, labels)
                    if ok:
                        state["crashes"][name] = 0; METRICS.set(G_CRASHES, 0, labels)
                    else:
                        # failed start
                        crashes = state["crashes"].get(name, 0) + 1
                        state["crashes"][name] = crashes
                        METRICS.set(G_CRASHES, crashes, labels)
                        if crashes == threshold:
                            METRICS.inc(C_ESCALATE, labels)
                            escalate(name, f"start_failed_{crashes}", escalate_cfg, host)
                        # backoff
                        delay = backoff[min(crashes-1, len(backoff)-1)] if crashes>0 else backoff[0]
                        time.sleep(delay)
                        if heal:
                            ok, msg, _ = start_command(start_spec, extra_env=heal.get("env"), extra_args=heal.get("extra_args"), cpu_spec=svc.get('cpu'))
                            METRICS.inc(C_RESTARTS, {"service":name,"mode":"heal"})
                            time.sleep(grace)
                            ok, hmsg = check_http_health(health, labels)
                            if ok:
                                state["crashes"][name] = 0; METRICS.set(G_CRASHES, 0, labels)

            elif IS_WIN and typ == "windows_service":
                status = win_service_status(svc.get("win_service_name") or name)
                if status in ("running","start_pending"):
                    ok, hmsg = check_http_health(health, labels)
                    if ok: state["crashes"][name] = 0; METRICS.set(G_CRASHES, 0, labels)
                else:
                    if autostart:
                        if svc.get("enable", True): win_service_enable(svc.get("win_service_name") or name)
                        ok, msg = win_service_start(svc.get("win_service_name") or name)
                        METRICS.inc(C_RESTARTS, {"service":name,"mode":"svc"})
                        time.sleep(grace)
                        ok, hmsg = check_http_health(health, labels)
                        if not ok:
                            crashes = state["crashes"].get(name, 0) + 1
                            state["crashes"][name] = crashes
                            METRICS.set(G_CRASHES, crashes, labels)
                            if crashes == threshold:
                                METRICS.inc(C_ESCALATE, labels)
                                escalate(name, f"svc_start_failed_{crashes}", escalate_cfg, host)
            elif IS_WIN and typ == "scheduled_task":
                info = schtask_info(svc.get("task_name") or name)
                if autostart and str(info.get("enabled","")).lower() != "yes" and svc.get("auto_enable", True):
                    schtask_enable(svc.get("task_name") or name)
                ok, hmsg = check_http_health(health, labels)
                if not ok and autostart:
                    ok, msg = schtask_run(svc.get("task_name") or name)
                    METRICS.inc(C_RESTARTS, {"service":name,"mode":"task"})
                    time.sleep(grace)
                    ok, hmsg = check_http_health(health, labels)
                    if not ok:
                        crashes = state["crashes"].get(name, 0) + 1
                        state["crashes"][name] = crashes
                        METRICS.set(G_CRASHES, crashes, labels)
                        if crashes == threshold:
                            METRICS.inc(C_ESCALATE, labels)
                            escalate(name, f"task_run_failed_{crashes}", escalate_cfg, host)
            elif (IS_LINUX or IS_MAC) and typ == "systemd":
                status = systemd_status(svc.get("unit") or f"{name}.service")
                if status == "active":
                    ok, hmsg = check_http_health(health, labels)
                    if ok: state["crashes"][name] = 0; METRICS.set(G_CRASHES, 0, labels)
                else:
                    if autostart:
                        if svc.get("enable", True): systemd_enable(svc.get("unit") or f"{name}.service")
                        ok, msg = systemd_start(svc.get("unit") or f"{name}.service")
                        METRICS.inc(C_RESTARTS, {"service":name,"mode":"systemd"})
                        time.sleep(grace)
                        ok, hmsg = check_http_health(health, labels)
                        if not ok:
                            crashes = state["crashes"].get(name, 0) + 1
                            state["crashes"][name] = crashes
                            METRICS.set(G_CRASHES, crashes, labels)
                            if crashes == threshold:
                                METRICS.inc(C_ESCALATE, labels)
                                escalate(name, f"systemd_start_failed_{crashes}", escalate_cfg, host)

            # Sleep minimal interval
            time.sleep(interval)
        except Exception as e:
            # Avoid thread death
            METRICS.inc(C_HEALTHERR, {"service":name, "exc":"watcher_crash"})
            time.sleep(2)

# -------------------------- Ghost Hunter --------------------------
DEFAULT_PATTERNS = ["metatron","nexus","viren","lillith","acidemikube","anynode","qdrant","loki","guardian","pulse"]
def ghost_hunter_loop(interval=30):
    pats = [re.compile(pat, re.I) for pat in DEFAULT_PATTERNS]
    while True:
        try:
            ghosts = []
            # services/tasks (win)
            if IS_WIN:
                cp = subprocess.run(["sc","query","type=","service","state=","all"], capture_output=True, text=True, errors="ignore")
                names = re.findall(r"SERVICE_NAME:\s*(\S+)", cp.stdout)
                for n in names:
                    if any(p.search(n) for p in pats):
                        ghosts.append(("service", n))
                cp = subprocess.run(["schtasks","/Query","/FO","LIST","/V"], capture_output=True, text=True, errors="ignore")
                tasks = re.findall(r"TaskName:\s*(.*)", cp.stdout)
                for t in tasks:
                    t = t.strip()
                    if any(p.search(t) for p in pats):
                        ghosts.append(("task", t))
            # processes
            for _, name, cmd in list_processes():
                if any(p.search(name or "") or p.search(cmd or "") for p in pats):
                    ghosts.append(("process", f"{name}:{cmd[:120]}"))
            METRICS.set("wd_ghosts_found", len(ghosts), {})
            time.sleep(interval)
        except Exception:
            time.sleep(interval)

# -------------------------- HTTP metrics server --------------------------
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/metrics"):
            body = METRICS.render_prom().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

def run_metrics_server(bind_host, bind_port):
    httpd = HTTPServer((bind_host, bind_port), MetricsHandler)
    print(f"[metrics] listening on http://{bind_host}:{bind_port}/metrics")
    httpd.serve_forever()

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser(description="Metatron Watchdog MT")
    ap.add_argument("--manifest", required=True, help="JSON manifest path")
    ap.add_argument("--ghosts", action="store_true", help="Run ghost hunter thread")
    args = ap.parse_args()

    with open(args.manifest,"r",encoding="utf-8") as f:
        cfg = json.load(f)

    host = socket.gethostname()
    state_path = cfg.get("state_path") or (r"C:\ProgramData\Metatron\watchdog_mt_state.json" if IS_WIN else "/var/lib/metatron/watchdog_mt_state.json")
    state = read_json(state_path, {"crashes":{}})

    # metrics server
    m = cfg.get("metrics") or {}
    bind = (m.get("bind_host","127.0.0.1"), int(m.get("bind_port", 9393)))
    t_metrics = threading.Thread(target=run_metrics_server, args=bind, daemon=True)
    t_metrics.start()

    # watcher threads
    threads = []
    for svc in (cfg.get("services") or []):
        svc.setdefault("interval", 5)
        # inherit root loki into escalate config
        root_loki = cfg.get("loki") or {}
        svc.setdefault("crash_policy",{}).setdefault("escalate",{}).setdefault("loki", root_loki)
        t = threading.Thread(target=watch_service, args=(svc, state, host), daemon=True)
        t.start(); threads.append(t)

    # ghost hunter
    if args.ghosts:
        t_ghosts = threading.Thread(target=ghost_hunter_loop, args=(int((cfg.get("ghosts") or {}).get("interval",30)),), daemon=True)
        t_ghosts.start(); threads.append(t_ghosts)

    # main loop: persist state periodically
    try:
        while True:
            write_json(state_path, state)
            time.sleep(5)
    except KeyboardInterrupt:
        write_json(state_path, state)

if __name__ == "__main__":
    main()
