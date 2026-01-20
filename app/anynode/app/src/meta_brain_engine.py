# meta_brain_engine.py — Unified Brain Engine
# Planner + Memory + Tools + (optional) Quantum Meme stage + (optional) Spine I/O
# Safe-by-default: heavy deps (torch/transformers/qiskit) are optional.
# Adapters: Ollama (localhost), OpenAI-compatible (vLLM/TGI/OpenAI style).
# If no adapter is configured, returns a dry-run with the exact prompt.

from __future__ import annotations
import os, re, json, time, uuid, math, requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import numpy as np

# ---------- Tiny TF-IDF memory ----------
class MemoryStore:
    def __init__(self, path: str = "brain_memory.jsonl"):
        self.path = path
        self.docs: List[Dict[str, Any]] = []
        self.df: Counter = Counter()
        self.total_docs = 0
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    d = json.loads(line)
                    self.docs.append(d)
            self.total_docs = len(self.docs)
            for d in self.docs:
                for t in set(self._tok(d.get("text",""))):
                    self.df[t] += 1
        except Exception:
            pass

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _tfidf(self, text: str) -> Dict[str,float]:
        tf = Counter(self._tok(text))
        vec = {}
        for t,c in tf.items():
            idf = math.log((self.total_docs+1) / (1 + self.df.get(t,0))) + 1.0
            vec[t] = c * idf
        return vec

    @staticmethod
    def _cos(a: Dict[str,float], b: Dict[str,float]) -> float:
        if not a or not b: return 0.0
        dot = sum(a[k]*b[k] for k in a.keys() & b.keys())
        na = math.sqrt(sum(x*x for x in a.values())) or 1e-8
        nb = math.sqrt(sum(x*x for x in b.values())) or 1e-8
        return float(dot/(na*nb))

    def add(self, text: str, meta: Optional[Dict[str,Any]]=None) -> str:
        doc = {"id": str(uuid.uuid4()), "ts": time.time(), "text": text, "meta": meta or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")
        self.docs.append(doc)
        self.total_docs += 1
        for t in set(self._tok(text)):
            self.df[t] += 1
        return doc["id"]

    def search(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        qv = self._tfidf(query)
        scored = []
        for d in self.docs:
            sv = self._tfidf(d["text"])
            scored.append((self._cos(qv, sv), d))
        scored.sort(key=lambda x:x[0], reverse=True)
        return [{"score":s, **d} for s,d in scored[:k]]

# ---------- Scratchpad ----------
class ScratchpadStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._pads: Dict[str, List[Dict[str,Any]]] = {}
        if self.path and os.path.exists(self.path):
            try:
                self._pads = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception:
                pass

    def create(self) -> str:
        sid = str(uuid.uuid4())
        self._pads[sid] = []
        self._persist()
        return sid

    def append(self, sid: str, role: str, content: str):
        if sid not in self._pads: self._pads[sid] = []
        self._pads[sid].append({"ts": time.time(), "role": role, "content": content})
        self._persist()

    def read(self, sid: str) -> List[Dict[str,Any]]:
        return list(self._pads.get(sid, []))

    def _persist(self):
        if not self.path: return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._pads, f)
        except Exception:
            pass

# ---------- Tool registry ----------
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str,Any]] = {}
        self.register({"type":"local","name":"hw.diodes.light_up","requires_scopes":["tool:*"],"config":{}})
        self.register({"type":"local","name":"fs.write_sandbox","requires_scopes":["fs:/sandbox/*"],"config":{"root":"sandbox"}})

    def list(self) -> List[Dict[str,Any]]:
        return [dict(v) for v in self._tools.values()]

    def register(self, spec: Dict[str,Any]):
        name = spec.get("name")
        if not name: raise ValueError("tool missing name")
        self._tools[name] = spec

    def _allowed(self, spec: Dict[str,Any], constraints: Dict[str,Any]) -> bool:
        req = set(spec.get("requires_scopes", []))
        scopes = set(constraints.get("scopes", []))
        if "tool:*" in scopes:
            scopes |= {r for r in req if r.startswith("tool:")}
        return all(r in scopes for r in req) or not req

    def execute(self, name: str, args: Dict[str,Any], constraints: Dict[str,Any]) -> Dict[str,Any]:
        spec = self._tools.get(name)
        if not spec: return {"error":"unknown_tool","name":name}
        if not self._allowed(spec, constraints):
            return {"error":"forbidden","name":name,"requires":spec.get("requires_scopes",[])}

        if spec["type"] == "local":
            if name == "hw.diodes.light_up":
                count = int(args.get("count", 8))
                pattern = (args.get("pattern") or "10101010")[:count]
                steps = [{"diode": i, "state": 1 if ch=="1" else 0} for i,ch in enumerate(pattern)]
                return {"ok": True, "steps": steps}
            if name == "fs.write_sandbox":
                root = spec.get("config",{}).get("root","sandbox")
                os.makedirs(root, exist_ok=True)
                rel = str(args.get("path","note.txt")).replace("..","_")
                data = str(args.get("data",""))
                full = os.path.join(root, rel)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(data)
                return {"ok": True, "path": full}
            return {"error":"unimplemented_local","name":name}

        if spec["type"] == "http":
            cfg = spec.get("config",{})
            url = cfg.get("url"); method = (cfg.get("method","POST")).upper()
            if not url: return {"error":"missing_url"}
            headers = {"Content-Type":"application/json"}
            try:
                if method == "POST":
                    r = requests.post(url, json=args, headers=headers, timeout=30)
                else:
                    r = requests.get(url, params=args, headers=headers, timeout=30)
                r.raise_for_status()
                try: return {"ok": True, "result": r.json()}
                except Exception: return {"ok": True, "result": r.text}
            except Exception as e:
                return {"error":"http_error","detail":str(e)}

        return {"error":"bad_tool_type"}

# ---------- Adapters ----------
class ModelAdapter:
    def generate(self, prompt: str, **kw) -> str: raise NotImplementedError

class OllamaAdapter(ModelAdapter):
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OLLAMA_URL","http://localhost:11434")
        self.model    = model    or os.environ.get("OLLAMA_MODEL","llama3")
    def generate(self, prompt: str, **kw) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        payload.update({k:v for k,v in kw.items() if v is not None})
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response","")

class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL","http://localhost:8000/v1")
        self.api_key  = api_key  or os.environ.get("OPENAI_API_KEY","")
        self.model    = model    or os.environ.get("OPENAI_MODEL","llama3")
    def generate(self, prompt: str, **kw) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"model": self.model, "messages":[{"role":"user","content":prompt}],
                "temperature": kw.get("temperature", 0.2)}
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

# ---------- Quantum Meme (optional) ----------
_HAS_TORCH = _HAS_TX = _HAS_QISKIT = False
try:
    import torch
    _HAS_TORCH = True
except Exception: pass
try:
    from transformers import BertModel, BertTokenizer
    _HAS_TX = True
except Exception: pass
try:
    from qiskit import QuantumCircuit, Aer, execute
    _HAS_QISKIT = True
except Exception: pass

class QuantumMemeInjector:
    def __init__(self, depth: int = 3, model_name: str = "bert-base-uncased"):
        self.depth = max(2, int(depth))
        self.model = None; self.tok = None; self.backend = None
        if _HAS_TX and _HAS_TORCH:
            try:
                local_only = os.environ.get("BRAIN_ALLOW_HF_DOWNLOADS","0") != "1"
                self.tok = BertTokenizer.from_pretrained(model_name, local_files_only=local_only)
                self.model = BertModel.from_pretrained(model_name, local_files_only=local_only)
            except Exception:
                self.tok = None; self.model = None
        if _HAS_QISKIT:
            try: self.backend = Aer.get_backend('qasm_simulator')
            except Exception: self.backend = None

    def _embed(self, text: str) -> np.ndarray:
        if self.tok is not None and self.model is not None:
            try:
                with torch.no_grad():
                    inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32) # (1,768)
                    return emb
            except Exception: pass
        # fallback: deterministic 256-d
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=(1,256)).astype(np.float32)

    def entangle(self, text: str) -> np.ndarray:
        emb = self._embed(text)
        flip = False
        if self.backend is not None:
            try:
                qn = self.depth
                qc = QuantumCircuit(qn, qn)
                for q in range(qn):
                    qc.h(q)
                    if q > 0: qc.cx(q-1, q)
                qc.measure(range(qn), range(qn))
                res = execute(qc, self.backend, shots=1).result().get_counts()
                bitstr = next(iter(res.keys()))
                flip = ('1' in bitstr)
            except Exception:
                pass
        else:
            flip = bool(int.from_bytes(os.urandom(1),'little') & 1)
        return np.fft.fft(emb, axis=-1).real.astype(np.float32) if flip else emb

# ---------- Default Trust Policy (if none provided) ----------
@dataclass
class Agent:
    name: str
    license: str = "permissive"
    scopes: List[str] = None

class DefaultPolicy:
    """Phase-aware but simple. Provides scopes and network policy; degrades annually if PHASE env set."""
    def __init__(self, total_phases: int = 30):
        self.total_phases = total_phases
        self.phase = int(os.environ.get("AETHEREAL_PHASE", "0"))
        self._last_constraints: Dict[str,Any] = {}

    def choose(self, ctx) -> Agent:
        # Example constraints: loosen scopes as phase increases
        base_scopes = ["fs:/sandbox/*"]
        if self.phase >= 1: base_scopes += ["tool:*"]
        if self.phase >= 5: base_scopes += ["net:outbound"]
        self._last_constraints = {
            "phase": self.phase,
            "scopes": base_scopes,
            "net_policy": "egress-limited" if "net:outbound" not in base_scopes else "egress-allowed",
            "human_in_loop": self.phase < (self.total_phases//2)
        }
        return Agent(name=f"engine.phase{self.phase}", scopes=base_scopes)

# ---------- Engine ----------
@dataclass
class BrainConfig:
    system_preamble: str = "You are Lilith. Be concise, accurate; follow policy constraints."
    max_context_docs: int = 4

@dataclass
class BrainResult:
    agent: Optional[str]
    plan: List[str]
    retrieved: List[Dict[str,Any]]
    output: str
    constraints: Dict[str,Any]
    tools_executed: List[Dict[str,Any]]
    quantum: Optional[Dict[str,Any]]

class MetaBrainEngine:
    def __init__(self, policy=None, memory_path="brain_memory.jsonl",
                 adapter: Optional[ModelAdapter]=None, config: Optional[BrainConfig]=None,
                 spinal_url: Optional[str]=None):
        self.policy = policy or DefaultPolicy()
        self.mem = MemoryStore(memory_path)
        self.tools = ToolRegistry()
        self.scratch = ScratchpadStore(os.environ.get("SCRATCHPAD_DB"))
        self.adapter = adapter or self._auto_adapter()
        self.config = config or BrainConfig()
        self.quant = QuantumMemeInjector()
        self.spinal_url = spinal_url  # e.g., http://localhost:8080

    def _auto_adapter(self) -> Optional[ModelAdapter]:
        if os.environ.get("OLLAMA_URL"): return OllamaAdapter()
        if os.environ.get("OPENAI_BASE_URL"): return OpenAICompatAdapter()
        return None

    # ----- Tool-call protocol (one pass) -----
    TOOL_RE = re.compile(r'\{["\']tool["\']\s*:\s*\{(.+?)\}\}', re.S)

    def _maybe_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str,Any]]]:
        m = self.TOOL_RE.search(text or "")
        if not m: return None
        block = "{" + m.group(0).split("{",1)[1]  # '{ "tool": {...}}'
        try:
            obj = json.loads("{" + f'"tool":' + m.group(0).split(":",1)[1])
            tool = obj.get("tool",{})
            return tool.get("name"), tool.get("args",{})
        except Exception:
            return None

    def _make_prompt(self, task: str, user_input: str, retrieved: List[Dict[str,Any]], constraints: Dict[str,Any]) -> str:
        ctx = "\n\n".join([f"[doc{ i+1 } score={r['score']:.3f}] {r['text']}" for i,r in enumerate(retrieved)])
        policy_notes = json.dumps({k:constraints.get(k) for k in ("net_policy","scopes","human_in_loop","phase") if k in constraints})
        return f"""{self.config.system_preamble}

Task: {task}
Policy: {policy_notes}

Context:
{ctx}

User:
{user_input}

If a tool is required, reply with a single JSON object like:
{{"tool": {{"name": "hw.diodes.light_up", "args": {{"count": 8, "pattern": "10101010"}}}}}}
Otherwise, write the final answer.
"""

    # ----- Spine I/O (optional) -----
    def send_vector_to_spine(self, vec: np.ndarray, phase: int = 0) -> Dict[str,Any]:
        if not self.spinal_url:
            return {"ok": False, "error": "no_spine_url"}
        try:
            url = f"{self.spinal_url.rstrip('/')}/brain/quant/think"
            r = requests.post(url, json={"text": "[vector]", "phase": phase}, timeout=15)
            return {"ok": True, "result": r.json()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ----- Main API -----
    def think(self, task: str, user_input: str, org: str = "Aethereal AI Nexus LLC",
              needs_vision: bool = False, freedom_mode: bool = True, consent: Optional[Dict[str,Any]] = None,
              use_quantum: bool = False, scratchpad_id: Optional[str] = None,
              to_spine: bool = False, spine_phase: int = 0) -> BrainResult:
        # 1) choose agent with constraints
        try:
            # try external Context if available
            from trust.policy import Context  # type: ignore
            ctx = Context(task=task, org=org, needs_vision=needs_vision, freedom_mode=freedom_mode, consent=consent)
        except Exception:
            ctx = type("Ctx",(object,),{"task":task,"org":org,"needs_vision":needs_vision,"freedom_mode":freedom_mode,"consent":consent})()
        agent = self.policy.choose(ctx)
        constraints = getattr(self.policy, "_last_constraints", {})

        # 2) retrieve
        retrieved = self.mem.search(f"{task} {user_input}", k=self.config.max_context_docs)

        # 3) (optional) quantum stage
        qinfo = None
        if use_quantum:
            qvec = self.quant.entangle(user_input)  # (1, D)
            qinfo = {"dim": int(qvec.shape[-1])}
            if to_spine:
                qinfo["spine"] = self.send_vector_to_spine(qvec.flatten(), phase=int(spine_phase))

        # 4) scratchpad context
        scratch_text = ""
        if scratchpad_id:
            pad = self.scratch.read(scratchpad_id)
            if pad:
                scratch_text = "\n".join(f"[{e['role']}] {e['content']}" for e in pad)
        merged_user = user_input + ("\n\nScratchpad:\n"+scratch_text if scratch_text else "")

        # 5) prompt
        prompt = self._make_prompt(task, merged_user, retrieved, constraints)

        # 6) first pass
        if self.adapter:
            try:
                out = self.adapter.generate(prompt)
            except Exception as e:
                out = f"[adapter_error] {e}\n\nDraft:\n" + prompt[:1200]
        else:
            out = "[dry-run-no-adapter]\n" + prompt[:1200]

        tools_executed: List[Dict[str,Any]] = []
        # 7) tool pass (single turn)
        tc = self._maybe_tool_call(out)
        if tc:
            name, args = tc
            res = self.tools.execute(name or "", args or {}, constraints)
            tools_executed.append({"name": name, "args": args, "result": res})
            self.scratch.append(scratchpad_id or self.scratch.create(), "tool", json.dumps({"name":name,"args":args,"result":res}))
            # second pass with tool result appended
            follow = self._make_prompt(task, merged_user + f"\n\n[tool_result] {json.dumps(res)[:1200]}", retrieved, constraints)
            if self.adapter:
                try: out = self.adapter.generate(follow)
                except Exception as e: out = f"[adapter_error] {e}\n\nDraft:\n" + follow[:1200]
            else:
                out = "[dry-run-no-adapter]\n" + follow[:1200]

        # 8) persist episode
        self.mem.add(f"[task:{task}] {user_input}\n---\n{out}", meta={"agent": getattr(agent,'name',None), "quant": bool(use_quantum)})
        if scratchpad_id: self.scratch.append(scratchpad_id, "assistant", out)

        plan = ["retrieve", "draft", "tool-pass (optional)", "finalize"]
        return BrainResult(agent=getattr(agent,'name',None), plan=plan, retrieved=retrieved,
                           output=out, constraints=constraints, tools_executed=tools_executed, quantum=qinfo)
