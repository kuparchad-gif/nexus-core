# src/lilith/mesh/cognikube_registry.py
from __future__ import annotations
import os, json, time, socket, threading, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import requests

DEFAULT_CFG = os.environ.get("COGNIKUBES_CFG", "config/cognikubes.json")
HEALTH_INTERVAL = float(os.environ.get("COGNIKUBES_HEALTH_S", "10"))
UDP_DISCOVERY = os.environ.get("COGNIKUBES_DISCOVERY", "1") == "1"

@dataclass
class Kube:
    id: str
    role: str
    url: str
    protocol: str = "http"          # http|grpc|udp (http default)
    health: str = "/health"
    auth_bearer: Optional[str] = None
    timeout_s: float = 10.0
    weight: float = 1.0
    capabilities: List[str] = field(default_factory=list)
    model_hint: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    last_status: str = "unknown"
    last_rtt_ms: float = -1.0
    last_checked: float = 0.0

class CogniKubeRegistry:
    def __init__(self, cfg_path: str = DEFAULT_CFG):
        self.cfg_path = cfg_path
        self._mt = 0.0
        self._lock = threading.RLock()
        self.kubes: Dict[str, Kube] = {}
        self._stop = threading.Event()
        self._load_cfg()
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def _load_cfg(self):
        with self._lock:
            try:
                mt = os.path.getmtime(self.cfg_path)
            except OSError:
                return
            if mt == self._mt:
                return
            data = json.load(open(self.cfg_path, "r", encoding="utf-8"))
            new_kubes: Dict[str, Kube] = {}
            for k in data.get("kubes", []):
                # env override for URL: CK_<ID>_URL
                kid = k["id"]
                env_url = os.environ.get(f"CK_{kid.upper()}_URL")
                url = env_url or k.get("url")
                new_kubes[kid] = Kube(
                    id=kid,
                    role=k.get("role","generic"),
                    url=url,
                    protocol=k.get("protocol","http"),
                    health=k.get("health","/health"),
                    auth_bearer=k.get("auth_bearer"),
                    timeout_s=float(k.get("timeout_s",10.0)),
                    weight=float(k.get("weight",1.0)),
                    capabilities=list(k.get("capabilities",[])),
                    model_hint=k.get("model_hint"),
                    tags=list(k.get("tags",[])),
                )
            self.kubes = new_kubes
            self._mt = mt

    def _health_loop(self):
        while not self._stop.is_set():
            self._load_cfg()
            self._probe_all()
            self._stop.wait(HEALTH_INTERVAL)

    def stop(self):
        self._stop.set()

    def _probe_all(self):
        for kube in list(self.kubes.values()):
            try:
                t0 = time.time()
                if kube.protocol == "http":
                    url = kube.url.rstrip("/") + kube.health
                    headers = {}
                    if kube.auth_bearer:
                        headers["Authorization"] = f"Bearer {kube.auth_bearer}"
                    r = requests.get(url, headers=headers, timeout=kube.timeout_s)
                    r.raise_for_status()
                    kube.last_status = "up"
                    kube.last_rtt_ms = (time.time()-t0)*1000.0
                elif kube.protocol == "udp" and UDP_DISCOVERY:
                    kube.last_status, kube.last_rtt_ms = self._udp_probe(kube)
                else:
                    kube.last_status = "unknown"
                    kube.last_rtt_ms = -1.0
            except Exception:
                kube.last_status = "down"
                kube.last_rtt_ms = -1.0
            kube.last_checked = time.time()

    def _udp_probe(self, kube: Kube) -> Tuple[str,float]:
        # expects URL like udp://host:port; sends "PULSE13?" and waits for any reply
        try:
            target = kube.url.split("://",1)[1]
            host, port = target.split(":")
            port = int(port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(kube.timeout_s)
            t0 = time.time()
            sock.sendto(b"PULSE13?", (host, port))
            _data, _addr = sock.recvfrom(1024)
            return "up", (time.time()-t0)*1000.0
        except Exception:
            return "down", -1.0

    def choose(self, role: Optional[str]=None, capability: Optional[str]=None, tags: Optional[List[str]]=None) -> Optional[Kube]:
        with self._lock:
            candidates = [k for k in self.kubes.values() if k.last_status=="up"]
            if role:
                candidates = [k for k in candidates if k.role==role]
            if capability:
                candidates = [k for k in candidates if capability in k.capabilities]
            if tags:
                st = set(tags)
                candidates = [k for k in candidates if st.issubset(set(k.tags))]
            if not candidates:
                return None
            # weighted random by weight; tiebreaker lower RTT
            total_w = sum(max(0.0, k.weight) for k in candidates) or 1.0
            r = random.random()*total_w
            upto = 0.0
            for k in sorted(candidates, key=lambda x: (x.last_rtt_ms if x.last_rtt_ms>0 else 1e9)):
                w = max(0.0, k.weight)
                if upto + w >= r:
                    return k
                upto += w
            return candidates[0]

    def list(self) -> List[Dict[str,Any]]:
        with self._lock:
            out = []
            for k in self.kubes.values():
                out.append({
                    "id": k.id, "role": k.role, "url": k.url, "protocol": k.protocol,
                    "status": k.last_status, "rtt_ms": k.last_rtt_ms, "capabilities": k.capabilities,
                    "weight": k.weight, "model_hint": k.model_hint, "tags": k.tags
                })
            return out

class CogniKubeClient:
    def __init__(self, registry: CogniKubeRegistry):
        self.r = registry

    def call_http(self, kube: Kube, path: str, payload: Dict[str,Any], method: str="POST", timeout: Optional[float]=None) -> Dict[str,Any]:
        url = kube.url.rstrip("/") + "/" + path.lstrip("/")
        headers = {"Content-Type":"application/json"}
        if kube.auth_bearer:
            headers["Authorization"] = f"Bearer {kube.auth_bearer}"
        t = timeout or kube.timeout_s
        if method.upper()=="POST":
            resp = requests.post(url, json=payload, headers=headers, timeout=t)
        else:
            resp = requests.get(url, params=payload, headers=headers, timeout=t)
        resp.raise_for_status()
        try: return resp.json()
        except Exception: return {"text": resp.text}

    def route(self, role: str, path: str, payload: Dict[str,Any], capability: Optional[str]=None, tags: Optional[List[str]]=None) -> Dict[str,Any]:
        kube = self.r.choose(role=role, capability=capability, tags=tags)
        if not kube:
            return {"error":"no_kube_available","role":role,"capability":capability,"tags":tags}
        if kube.protocol != "http":
            return {"error":"protocol_not_supported_yet","protocol": kube.protocol, "kube": kube.id}
        return self.call_http(kube, path, payload)

# convenience singleton
_registry_singleton: Optional[CogniKubeRegistry] = None
_client_singleton: Optional[CogniKubeClient] = None

def registry() -> CogniKubeRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = CogniKubeRegistry()
    return _registry_singleton

def client() -> CogniKubeClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = CogniKubeClient(registry())
    return _client_singleton
