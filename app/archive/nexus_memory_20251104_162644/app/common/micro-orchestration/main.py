
import os, json, time, asyncio, pathlib, shutil, hashlib, platform
from nats.aio.client import Client as NATS
from providers import make_provider
from rules import CATEGORIES, score_by_rules

def getenv_map():
    keys  =  ["PROVIDER","LMSTUDIO_URL","LMSTUDIO_MODEL","OLLAMA_URL","OLLAMA_MODEL","VLLM_URL","VLLM_MODEL",
            "SRC_ROOT","DEST_ROOT","DRY_RUN","MAX_BYTES","INCLUDE_EXTS","IGNORE_DIRS","NATS_URL","TENANT","PROJECT",
            "DO_METADATA","SUPPORT_MODE","SIDECAR_EXT","SUPPORT_DIR","CONFIDENCE_MIN","ALLOW_SAME_PATH"]
    return {k: os.getenv(k) for k in keys}

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def redacted_head(path: pathlib.Path, max_bytes: int) -> str:
    try:
        with open(path, "rb") as f:
            blob  =  f.read(max_bytes)
        return blob.decode("utf-8", errors = "ignore")
    except Exception:
        return ""

def classify_llm(provider, path: pathlib.Path, header: str) -> tuple[str, float, str]:
    if provider is None or not header.strip():
        return "", 0.0, "no-llm"
    cats  =  CATEGORIES
    schema_hint  =  {"category": "one of CATEGORIES", "confidence": "0..1", "reason": "string"}
    prompt  =  f"""
You are a repository file organizer for an AI OS of micro-kubes.
Pick one best target category for the file based on path and header. Copy-only mapping; no edits.
Return STRICT JSON object with keys exactly: category, confidence, reason.

File path: {str(path)}
Header: <<<{header[:2000]}>>>
Categories: {cats}
Examples of mapping:
- NATS or compose configs -> "infra"
- mind.obs.event / layer logic -> "Reasoner-Conscious/apps" or "Reasoner-Subconscious/apps"
- upper.plan/digest -> "UpperCognition/apps"
- cogswitch/memory.write/digest -> "SubconSwitch/apps"
- cogbridge ask/tell -> "CognitionBridge/apps"
- React/TSX front-end -> "frontends/portal"

Return only JSON like: {json.dumps(schema_hint)}
"""
    try:
        txt  =  provider.classify(prompt)
        start  =  txt.find("{"); end  =  txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj  =  json.loads(txt[start:end+1])
            cat  =  obj.get("category",""); conf  =  float(obj.get("confidence",0)); reason  =  obj.get("reason","llm")
            if cat in cats and 0 < =  conf < =  1:
                return cat, conf, reason
    except Exception as e:
        print("[categorizer] LLM error:", e)
    return "", 0.0, "llm-fail"

def walk_files(root: pathlib.Path, include_exts: set[str], ignore_dirs: set[str]):
    ignore_dirs  =  {d.lower() for d in ignore_dirs}
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name.lower() in ignore_dirs:
                continue
            else:
                continue
        if include_exts and p.suffix.lower() not in include_exts:
            continue
        yield p

def map_target(category: str, path: pathlib.Path):
    if category.endswith("/apps"):
        return f"CogniKubes/{category}/{path.name}"
    if category in ("protocols","infra","common"):
        return f"{category}/{path.name}"
    if category.startswith("frontends/"):
        return f"{category}/{path.name}"
    return f"common/{path.name}"

def ensure_dest_root(src_root: pathlib.Path, dest_root: pathlib.Path, allow_same: bool) -> pathlib.Path:
    if src_root.resolve() == dest_root.resolve() and not allow_same:
        rerouted  =  pathlib.Path(str(dest_root) + (".organized" if platform.system() =  = "Windows" else "_organized"))
        print(f"[categorizer] SRC_ROOT == DEST_ROOT; rerouting DEST_ROOT -> {rerouted}")
        return rerouted
    return dest_root

def write_metadata(meta_mode: str, sidecar_ext: str, support_dir: pathlib.Path, dest_root: pathlib.Path, dst_path: pathlib.Path, meta: dict):
    if meta_mode == "sidecar":
        sidecar  =  dst_path.with_suffix(dst_path.suffix + sidecar_ext)
        sidecar.parent.mkdir(parents = True, exist_ok = True)
        with open(sidecar, "w", encoding = "utf-8") as f:
            json.dump(meta, f, indent = 2)
        return str(sidecar)
    rel  =  dst_path.relative_to(dest_root)
    side  =  (support_dir / rel).with_suffix(rel.suffix + ".json") if support_dir else dst_path.with_suffix(dst_path.suffix + ".json")
    side.parent.mkdir(parents = True, exist_ok = True)
    with open(side, "w", encoding = "utf-8") as f:
        json.dump(meta, f, indent = 2)
    return str(side)

async def main():
    env  =  getenv_map()
    provider  =  make_provider(env)
    SRC_ROOT  =  pathlib.Path(env.get("SRC_ROOT") or "/src")
    DEST_ROOT  =  pathlib.Path(env.get("DEST_ROOT") or "/out")
    DRY_RUN  =  (env.get("DRY_RUN","true").lower() =  = "true")
    MAX_BYTES  =  int(env.get("MAX_BYTES") or "65536")
    include_exts  =  set([e.strip().lower() for e in (env.get("INCLUDE_EXTS") or "").split(",") if e.strip()])
    ignore_dirs  =  set([d.strip() for d in (env.get("IGNORE_DIRS") or "").split(",") if d.strip()])
    DO_METADATA  =  (env.get("DO_METADATA","true").lower() =  = "true")
    SUPPORT_MODE  =  (env.get("SUPPORT_MODE") or "sidecar").lower()
    SIDECAR_EXT  =  env.get("SIDECAR_EXT") or ".meta.json"
    SUPPORT_DIR  =  env.get("SUPPORT_DIR") or "_support"
    CONF_MIN  =  float(env.get("CONFIDENCE_MIN") or "0.0")
    ALLOW_SAME  =  (env.get("ALLOW_SAME_PATH","false").lower() =  = "true")

    DEST_ROOT  =  ensure_dest_root(SRC_ROOT, DEST_ROOT, ALLOW_SAME)

    plan  =  []
    for p in walk_files(SRC_ROOT, include_exts, ignore_dirs):
        head  =  redacted_head(p, MAX_BYTES)
        rule_cat, rule_conf, rule_reason  =  score_by_rules(p, head)
        llm_cat, llm_conf, llm_reason  =  classify_llm(provider, p, head)

        if llm_conf > rule_conf + 0.05:
            cat, conf, why  =  llm_cat, llm_conf, f"llm:{llm_reason}"
        else:
            cat, conf, why  =  rule_cat, rule_conf, f"rule:{rule_reason}"

        if conf < CONF_MIN:
            dst  =  f"_unclassified/{p.name}"
        else:
            dst  =  map_target(cat, p)

        plan.append({"src": str(p), "dst": dst, "category": cat, "confidence": conf, "why": why})

    state_dir  =  pathlib.Path("var/state"); state_dir.mkdir(parents = True, exist_ok = True)
    plan_path  =  state_dir / "plan.json"
    with open(plan_path, "w", encoding = "utf-8") as f:
        json.dump({"created_ts": int(time.time()), "count": len(plan), "plan": plan}, f, indent = 2)

    copied  =  0; errors  =  0; metas  =  0
    support_dir  =  (DEST_ROOT / SUPPORT_DIR) if SUPPORT_MODE =  = "tree" else None
    for item in plan:
        src  =  pathlib.Path(item["src"])
        dst  =  DEST_ROOT / item["dst"].lstrip("/")
        if DRY_RUN:
            continue
        try:
            dst.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(src, dst)
            copied + =  1

            if DO_METADATA:
                meta  =  {
                    "src": str(src),
                    "dst": str(dst),
                    "category": item["category"],
                    "confidence": item["confidence"],
                    "why": item["why"],
                    "sha256": sha256_file(src),
                    "ts": int(time.time())
                }
                if SUPPORT_MODE == "tree":
                    support_dir.mkdir(parents = True, exist_ok = True)
                    write_metadata(SUPPORT_MODE, SIDECAR_EXT, support_dir, DEST_ROOT, dst, meta)
                else:
                    write_metadata("sidecar", SIDECAR_EXT, None, DEST_ROOT, dst, meta)
                metas + =  1
        except Exception as e:
            print("[categorizer] copy error:", src, "->", dst, e)
            errors + =  1

    nats_url  =  os.getenv("NATS_URL")
    if nats_url:
        TENANT  =  os.getenv("TENANT","AETHEREAL"); PROJECT  =  os.getenv("PROJECT","METANET")
        PUB_PLAN  =  f"nexus.{TENANT}.{PROJECT}.categorizer.tell.plan"
        PUB_DONE  =  f"nexus.{TENANT}.{PROJECT}.categorizer.tell.copied"
        nc  =  NATS()
        try:
            await nc.connect(servers = [nats_url], allow_reconnect = True, max_reconnect_attempts = -1, reconnect_time_wait = 2)
            await nc.publish(PUB_PLAN, json.dumps({"plan_path": str(plan_path), "count": len(plan)}).encode())
            if not DRY_RUN:
                await nc.publish(PUB_DONE, json.dumps({"copied":copied,"errors":errors,"metas":metas,"dest_root":str(DEST_ROOT)}).encode())
        except Exception as e:
            print("[categorizer] NATS notify error:", e)
        finally:
            try:
                await nc.close()
            except Exception:
                pass

if __name__ =  = "__main__":
    asyncio.run(main())
