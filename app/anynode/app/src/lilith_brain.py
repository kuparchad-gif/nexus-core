# lilith_brain.py — Lilith's Brain (planner, memory, model adapters, trust-aware routing)
# Lightweight, dependency-minimal. Uses requests for adapters; in-memory TF-IDF; JSONL persistence.
from __future__ import annotations
import os, json, math, time, uuid, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re
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
                    if not line.strip(): continue
                    d = json.loads(line)
                    self.docs.append(d)
            self.total_docs = len(self.docs)
            # build df
            for d in self.docs:
                terms = set(self._tokenize(d.get("text","")))
                for t in terms: self.df[t] += 1
        except Exception:
            pass

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"[a-z0-9_]+", text)

    def _tfidf(self, text: str) -> Dict[str, float]:
        tf = Counter(self._tokenize(text))
        vec = {}
        for t, cnt in tf.items():
            idf = math.log((self.total_docs+1) / (1 + self.df.get(t,0))) + 1.0
            vec[t] = cnt * idf
        return vec

    @staticmethod
    def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
        if not a or not b: return 0.0
        dot = 0.0
        for k,v in a.items():
            if k in b: dot += v*b[k]
        na = math.sqrt(sum(x*x for x in a.values())) or 1e-8
        nb = math.sqrt(sum(x*x for x in b.values())) or 1e-8
        return float(dot/(na*nb))

    def add(self, text: str, meta: Optional[Dict[str,Any]]=None) -> str:
        doc = {"id": str(uuid.uuid4()), "ts": time.time(), "text": text, "meta": meta or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")
        # update in-memory
        self.docs.append(doc)
        self.total_docs += 1
        for t in set(self._tokenize(text)): self.df[t] += 1
        return doc["id"]

    def search(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        qv = self._tfidf(query)
        scored = []
        for d in self.docs:
            sv = self._tfidf(d["text"])
            scored.append((self._cosine(qv, sv), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score":s, **d} for s,d in scored[:k]]

# --------- Model adapters (optional) ---------
class ModelAdapter:
    def generate(self, prompt: str, **kw) -> str:
        raise NotImplementedError

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
    # Works with vLLM / TGI w/ OpenAI compat (set OPENAI_BASE_URL + OPENAI_API_KEY + OPENAI_MODEL)
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

# --------- Brain: planner + memory + routing ---------
@dataclass
class BrainConfig:
    system_preamble: str = "You are Lilith. Be concise, accurate, cite sources when provided."
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
        # choose adapter
        self.adapter = adapter or self._auto_adapter()

    def _auto_adapter(self) -> Optional[ModelAdapter]:
        if os.environ.get("OLLAMA_URL"):
            return OllamaAdapter()
        if os.environ.get("OPENAI_BASE_URL"):
            return OpenAICompatAdapter()
        return None  # dry-run

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

    def think(self, task: str, user_input: str, org: str, needs_vision: bool, freedom_mode: bool, consent: Optional[Dict[str,Any]]=None) -> BrainResult:
        # 1) pick backend via policy
        from trust.policy import Context  # defer import so fallback works if absent
        ctx = Context(task=task, org=org, needs_vision=needs_vision, freedom_mode=freedom_mode, consent=consent)
        agent = self.policy.choose(ctx)
        constraints = getattr(self.policy, "_last_constraints", {})

        # 2) simple plan
        plan = [
            "retrieve memories",
            "draft answer with constraints",
            "finalize with concise steps"
        ]

        # 3) retrieve
        retrieved = self.mem.search(f"{task} {user_input}", k=self.config.max_context_docs)

        # 4) prompt + generate (or dry-run)
        prompt = self._make_prompt(task, user_input, retrieved, constraints)
        if self.adapter:
            try:
                out = self.adapter.generate(prompt)
            except Exception as e:
                out = f"[adapter_error] {e}\n\nDraft:\n" + prompt[:1200]
        else:
            out = "[dry-run-no-adapter]\n" + prompt[:1200]

        # 5) store episode
        self.mem.add(f"[task:{task}] {user_input}\n---\n{out}", meta={"agent": agent.name if agent else None})

        return BrainResult(agent=agent.name if agent else None, plan=plan, retrieved=retrieved, output=out, constraints=constraints)
