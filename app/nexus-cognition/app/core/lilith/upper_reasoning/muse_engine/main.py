# C:\Projects\Stacks\nexus-metatron\backend\services\muse_engine\main.py
# Muse Engine (Creative/Artistic Subsystem)
# - Subscribes to cog.obs.event (pure observations)
# - Translates emotion/mode/frequency into artistic artifacts (text prompts or poems)
# - Can call Ollama/vLLM if available, otherwise uses deterministic procedural generation
import os, json, time, asyncio, random, hashlib, requests

from nats.aio.client import Client as NATS

TENANT  = os.getenv("TENANT","AETHEREAL")
PROJECT = os.getenv("PROJECT","METANET")
NATS_URL= os.getenv("NATS_URL","nats://nats:4222")

OBS_EVENT  = f"nexus.{TENANT}.{PROJECT}.cog.obs.event"
ART_OUT    = f"nexus.{TENANT}.{PROJECT}.muse.artifact"

OLLAMA_URL = os.getenv("OLLAMA_URL","http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","llama3.1")
VLLM_URL   = os.getenv("VLLM_URL","http://vllm:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL","facebook/opt-125m")
USE_LLM    = os.getenv("MUSE_USE_LLM","false").lower()=="true"

# Emotion-to-style grammar (deterministic defaults)
STYLE = {
  "curiosity":   {"tempo":"andante","palette":["azure","silver"],"structure":"questioned couplet","harmonic":"f369"},
  "chaos":       {"tempo":"presto","palette":["crimson","obsidian"],"structure":"fractured free verse","harmonic":"f6"},
  "mystery":     {"tempo":"adagio","palette":["violet","indigo"],"structure":"elliptical tercets","harmonic":"f9"},
  "wonder":      {"tempo":"allegro","palette":["gold","sapphire"],"structure":"expanding quatrains","harmonic":"f3"},
  "imagination": {"tempo":"moderato","palette":["emerald","amber"],"structure":"what-if cascades","harmonic":"f369"},
  "abstract":    {"tempo":"largo","palette":["slate","ivory"],"structure":"axiomatic couplets","harmonic":"f432"}
}

def _seed(obs_id:str):
    # Stable seed from observation id
    h = int(hashlib.sha256(obs_id.encode()).hexdigest(),16) % (2**32-1)
    random.seed(h)

def _procedural_poem(mode:str, summary:str, frequency:str)->str:
    _seed(summary + frequency + mode)
    s = STYLE.get(mode, STYLE["curiosity"])
    lines = []
    lines.append(f"[tempo:{s['tempo']}] [harmonic:{s['harmonic']}] [palette:{'/'.join(s['palette'])}]")
    lines.append(f"{mode.upper()} // {summary or 'untitled observation'}")
    vocab = {
      "curiosity": ["why", "how", "trace", "edge", "opening"],
      "chaos": ["shatter", "swerve", "tilt", "spark", "cascade"],
      "mystery": ["veil", "echo", "cipher", "shadow", "key"],
      "wonder": ["glow", "lift", "horizon", "bloom", "halo"],
      "imagination": ["what-if", "bridge", "mirror", "gate", "prototype"],
      "abstract": ["form", "map", "lattice", "axiom", "limit"]
    }[mode if mode in ["curiosity","chaos","mystery","wonder","imagination","abstract"] else "curiosity"]
    for i in range(4):
        a, b = random.sample(vocab, 2)
        lines.append(f"{a} ↔ {b}")
    return "\n".join(lines)

def _ollama_generate(prompt:str)->str:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=30)
        if r.ok:
            data = r.json()
            return data.get("response","").strip()
    except Exception:
        pass
    return ""

def _vllm_generate(prompt:str)->str:
    try:
        r = requests.post(VLLM_URL, json={"model": VLLM_MODEL, "messages":[{"role":"user","content": prompt}]}, timeout=30)
        if r.ok:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return ""

def make_prompt(obs:dict)->str:
    mode = obs.get("mode","curiosity")
    s = STYLE.get(mode, STYLE["curiosity"])
    return (
        "Compose a short artistic reflection that encodes emotion into structure.\n"
        f"Mode: {mode}\n"
        f"Tempo: {s['tempo']}\nPalette: {', '.join(s['palette'])}\n"
        f"Structure: {s['structure']}\nHarmonic lane: {s['harmonic']}\n"
        f"Observation summary: {obs.get('summary','(none)')}\n"
        "Use 6-10 lines. Begin with a tag line of the form "
        "[tempo:...][harmonic:...][palette:...]. Keep it tasteful."
    )

async def main():
    nc = NATS(); await nc.connect(servers=[NATS_URL])

    async def handler(msg):
        try:
            obs = json.loads(msg.data)
        except Exception:
            return
        mode = obs.get("mode","curiosity")
        frequency = obs.get("frequency","f369")
        summary = obs.get("summary","")

        artifact = ""
        if USE_LLM:
            prompt = make_prompt(obs)
            artifact = _ollama_generate(prompt) or _vllm_generate(prompt)

        if not artifact:
            artifact = _procedural_poem(mode, summary, frequency)

        out = {
            "artifact_id": obs["obs_id"],
            "ts": int(time.time()),
            "tenant": TENANT,
            "project": PROJECT,
            "mode": mode,
            "frequency": frequency,
            "source_task": obs.get("task_id",""),
            "artifact_text": artifact
        }
        await nc.publish(ART_OUT, json.dumps(out).encode())
        print(json.dumps({"event":"muse_artifact", "id": out["artifact_id"], "mode": mode}))

    await nc.subscribe(OBS_EVENT, cb=handler)
    await asyncio.Event().wait()

if __name__=="__main__":
    asyncio.run(main())
