# lilith_brain.py — Lilith's Brain (planner, memory, tools, scratchpad, adapters)
from __future__ import annotations
import os, json, math, time, uuid, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import requests

# --------- Tiny TF-IDF memory store ---------
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
                    if not line: 
                        continue
                    d = json.loads(line)
                    self.docs.append(d)
            self.total_docs = len(self.docs)
            for d in self.docs:
                for t in set(self._tokenize(d.get("text",""))):
                    self.df[t] += 1
        except Exception:
            pass

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _tfidf(self, text: str) -> Dict[str, float]:
        tf = Counter(self._tokenize(text))
        vec = {}
        for t, cnt in tf.items():
            idf = math.log((self.total_docs+1) / (1 + self.df.get(t,0))) + 1.0
            vec[t] = cnt * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
        if not a or not b: 
            return 0.0
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
        for t in set(self._tokenize(text)):
            self.df[t] += 1
        return doc["id"]

    def search(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        qv = self._tfidf(query)
        scored = []
        for d in self.docs:
            sv = self._tfidf(d["text"])
            scored.append((self._cosine(qv, sv), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score":s, **d} for s,d in scored[:k]]

# --------- Scratchpad (ephemeral with optional persistence) ---------
class ScratchpadStore:
    def __init__(self, path: str = None):
        self.path = path
        self._pads: Dict[str, List[Dict[str,Any]]] = {}
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._pads = data
            except Exception:
                pass

    def create(self) -> str:
        sid = str(uuid.uuid4())
        self._pads[sid] = []
        self._persist()
        return sid

    def append(self, sid: str, role: str, content: str):
        if sid not in self._pads:
            self._pads[sid] = []
        self._pads[sid].append({"ts": time.time(), "role": role, "content": content})
        self._persist()

    def read(self, sid: str) -> List[Dict[str,Any]]:
        return list(self._pads.get(sid, []))

    def _persist(self):
        if not self.path: 
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._pads, f)
        except Exception:
            pass

# --------- Tool Registry ---------
class ToolRegistry:
    """
    Tools are lightweight function descriptors:
    {
      "type": "local" | "http",
      "name": "hw.diodes.light_up",
      "requires_scopes": ["tool:*"],
      "config": {...}
    }
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str,Any]] = {}
        # Built-ins
        self.register({"type":"local","name":"hw.diodes.light_up","requires_scopes":["tool:*"],"config":{}})
        self.register({"type":"local","name":"fs.write_sandbox","requires_scopes":["fs:/sandbox/*"],"config":{"root":"sandbox"}})

    def list(self) -> List[Dict[str,Any]]:
        return [dict(v) for v in self._tools.values()]

    def register(self, spec: Dict[str,Any]):
        name = spec.get("name")
        if not name: 
            raise ValueError("tool missing name")
        self._tools[name] = spec

    def _allowed(self, spec: Dict[str,Any], constraints: Dict[str,Any]) -> bool:
        req = set(spec.get("requires_scopes", []))
        scopes = set(constraints.get("scopes", []))
        if "tool:*" in scopes:
            scopes |= {r for r in req if r.startswith("tool:")}
        return all(r in scopes for r in req) or not req

    def execute(self, name: str, args: Dict[str,Any], constraints: Dict[str,Any]) -> Dict[str,Any]:
        spec = self._tools.get(name)
        if not spec:
            return {"error":"unknown_tool","name":name}
        if not self._allowed(spec, constraints):
            return {"error":"forbidden","name":name,"requires":spec.get("requires_scopes",[])}

        if spec["type"] == "local":
            if name == "hw.diodes.light_up":
                count = int(args.get("count", 8))
                pattern = (args.get("pattern") or "10101010")[:count]
                steps = [{"diode": i, "state": 1 if ch=="1" else 0} for i, ch in enumerate(pattern)]
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
            if not url:
                return {"error":"missing_url"}
            headers = {"Content-Type":"application/json"}
            try:
                if method == "POST":
                    r = requests.post(url, json=args, headers=headers, timeout=30)
                else:
                    r = requests.get(url, params=args, headers=headers, timeout=30)
                r.raise_for_status()
                try:
                    return {"ok": True, "result": r.json()}
                except Exception:
                    return {"ok": True, "result": r.text}
            except Exception as e:
                return {"error":"http_error","detail":str(e)}

        return {"error":"bad_tool_type"}

# --------- Adapters ---------
class ModelAdapter:
    def generate(self, prompt: str, **kw) -> str: raise NotImplementedError

class OllamaAdapter(ModelAdapter):
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OLLAMA_URL","http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL","llama3")
    def generate(self, prompt: str, **kw) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        payload.update({k:v for k,v in kw.items() if v is not None})
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response","")

class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL","http://localhost:8000/v1")
        self.api_key  = api_key  or os.environ.get("OPENAI_API_KEY","")
        self.model    = model    or os.environ.get("OPENAI_MODEL","llama3")
    def generate(self, prompt: str, **kw) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"model": self.model, "messages":[{"role":"user","content":prompt}], "temperature": kw.get("temperature", 0.2)}
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

# --------- Brain ---------
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

class LilithBrain:
    def __init__(self, policy, memory_path: str = "brain_memory.jsonl", adapter: Optional[ModelAdapter]=None, config: Optional[BrainConfig]=None):
        self.policy = policy
        self.mem = MemoryStore(memory_path)
        self.config = config or BrainConfig()
        self.adapter = adapter or self._auto_adapter()
        self.tools = ToolRegistry()
        self.scratch = ScratchpadStore(os.environ.get("SCRATCHPAD_DB"))

    def _auto_adapter(self) -> Optional[ModelAdapter]:
        if os.environ.get("OLLAMA_URL"):
            return OllamaAdapter()
        if os.environ.get("OPENAI_BASE_URL"):
            return OpenAICompatAdapter()
        return None

    def _make_prompt(self, task: str, user_input: str, retrieved: List[Dict[str,Any]], constraints: Dict[str,Any]) -> str:
        ctx = "\n\n".join([f"[doc{ i+1 } score={r['score']:.3f}] {r['text']}" for i,r in enumerate(retrieved)])
        policy_notes = json.dumps({k:constraints.get(k) for k in ("net_policy","scopes","human_in_loop") if k in constraints})
        return f"""{self.config.system_preamble}

Task: {task}
Policy: {policy_notes}

Context:
{ctx}

User:
{user_input}

Answer with clear steps and keep to the policy.
"""

    def think(self, task: str, user_input: str, org: str, needs_vision: bool, freedom_mode: bool, consent: Optional[Dict[str,Any]]=None, scratchpad_id: Optional[str]=None) -> BrainResult:
        from trust.policy import Context  # defer import so fallback works
        ctx = Context(task=task, org=org, needs_vision=needs_vision, freedom_mode=freedom_mode, consent=consent)
        agent = self.policy.choose(ctx)
        constraints = getattr(self.policy, "_last_constraints", {})

        plan = ["retrieve memories","draft under constraints","tool-optional","finalize"]
        retrieved = self.mem.search(f"{task} {user_input}", k=self.config.max_context_docs)

        scratch_text = ""
        if scratchpad_id:
            pad = self.scratch.read(scratchpad_id)
            if pad:
                scratch_text = "\n".join(f"[{e['role']}] {e['content']}" for e in pad)

        prompt = self._make_prompt(task, user_input + ("\n\nScratchpad:\n"+scratch_text if scratch_text else ""), retrieved, constraints)
        if self.adapter:
            try:
                out = self.adapter.generate(prompt)
            except Exception as e:
                out = f"[adapter_error] {e}\n\nDraft:\n" + prompt[:1200]
        else:
            out = "[dry-run-no-adapter]\n" + prompt[:1200]

        if scratchpad_id:
            self.scratch.append(scratchpad_id, "assistant", out)

        self.mem.add(f"[task:{task}] {user_input}\n---\n{out}", meta={"agent": agent.name if agent else None})

        return BrainResult(agent=agent.name if agent else None, plan=plan, retrieved=retrieved, output=out, constraints=constraints)
