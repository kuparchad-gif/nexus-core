# services/common/verify.py
import os, re, typing as T
import httpx

MIN_SOURCES = int(os.getenv("VERIFY_MIN_SOURCES","2"))
MIN_COVERAGE = float(os.getenv("VERIFY_MIN_COVERAGE","0.7"))
MAX_AGE_DAYS = int(os.getenv("VERIFY_MAX_AGE_DAYS","180"))
ALLOWLIST = [s.strip() for s in os.getenv("DOMAIN_ALLOWLIST","").split(",") if s.strip()]
BACKEND = os.getenv("VERIFIER_BACKEND","mnli").lower()
MNLI_MODEL_NAME = os.getenv("MNLI_MODEL_NAME","roberta-large-mnli")

_nli_pipe = None
def get_nli_pipe():
    global _nli_pipe
    if _nli_pipe is None and BACKEND == "mnli":
        from transformers import pipeline
        _nli_pipe = pipeline("text-classification", model=MNLI_MODEL_NAME, return_all_scores=True, device_map="auto")
    return _nli_pipe

def split_claims(text: str) -> T.List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    claims = [p.strip() for p in parts if p.strip()]
    return [c for c in claims if not c.endswith("?")]

def looks_official(url: str) -> bool:
    host = url.split("/")[2] if "://" in url else url
    return any(token in host for token in ALLOWLIST)

async def search_web(query: str) -> T.List[dict]:
    out = []
    async with httpx.AsyncClient(timeout=20) as c:
        tkey = os.getenv("TAVILY_API_KEY")
        if tkey:
            try:
                r = await c.post("https://api.tavily.com/search", json={"api_key": tkey, "query": query, "search_depth":"advanced"})
                r.raise_for_status(); data = r.json()
                for it in data.get("results", []):
                    out.append({"url": it.get("url",""), "title": it.get("title",""), "snippet": it.get("content","")[:500], "published": it.get("published_date") or ""})
            except Exception: pass
        bkey = os.getenv("BRAVE_API_KEY")
        if bkey and len(out)<3:
            try:
                r = await c.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": 5}, headers={"X-Subscription-Token": bkey})
                r.raise_for_status(); data = r.json()
                for it in data.get("web",{}).get("results",[]):
                    out.append({"url": it.get("url",""), "title": it.get("title",""), "snippet": it.get("description","")[:500], "published": it.get("page_age") or ""})
            except Exception: pass
        mkey = os.getenv("BING_SEARCH_V7_SUBSCRIPTION_KEY")
        if mkey and len(out)<3:
            try:
                r = await c.get("https://api.bing.microsoft.com/v7.0/search", params={"q":query, "count":5}, headers={"Ocp-Apim-Subscription-Key": mkey})
                r.raise_for_status(); data = r.json()
                for it in data.get("webPages",{}).get("value",[]):
                    out.append({"url": it.get("url",""), "title": it.get("name",""), "snippet": it.get("snippet","")[:500], "published": it.get("dateLastCrawled") or ""})
            except Exception: pass
    seen, deduped = set(), []
    for r in out:
        u = r.get("url","")
        if not u or u in seen: continue
        seen.add(u); deduped.append(r)
    return deduped[:10]

def numeric_fingerprint(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(n) for n in nums]

def numeric_consistent(a: str, b: str, tol=0.05) -> bool:
    na, nb = numeric_fingerprint(a), numeric_fingerprint(b)
    if not na or not nb: return True
    N = min(len(na), len(nb))
    if N == 0: return True
    for i in range(N):
        va, vb = na[i], nb[i]
        if va == 0 and vb == 0: continue
        if va == 0 or vb == 0: return False
        if abs(va-vb)/max(abs(va),abs(vb)) > tol: return False
    return True

async def nli_entails(premise: str, hypothesis: str) -> float:
    if BACKEND == "mnli":
        pipe = get_nli_pipe()
        res = pipe({"text": premise, "text_pair": hypothesis})
        score = 0.0
        for item in res[0]:
            if item["label"].lower().startswith("entail"):
                score = float(item["score"]); break
        return score
    else:
        prem = (premise or "").lower().split(); hyp = (hypothesis or "").lower().split()
        if not hyp: return 0.0
        overlap = len(set(prem) & set(hyp)) / max(1, len(set(hyp)))
        return max(0.0, min(1.0, overlap))

async def verify_answer(question: str, answer: str) -> dict:
    claims = split_claims(answer)
    if not claims:
        return {"verified": False, "score": 0.0, "reason": "no_claims", "evidence": []}

    core_query = (question or "").strip() or claims[0]
    results = await search_web(core_query)
    if not results:
        return {"verified": False, "score": 0.0, "reason": "no_evidence_found", "evidence": []}

    filtered = [r for r in results if looks_official(r.get("url","")) or any(seg in r.get("url","") for seg in ["/docs","/developer","/learn","/support"])]
    if len(filtered) < MIN_SOURCES:
        filtered = results

    evidence = []; covered = 0
    for claim in claims:
        best = None; best_score = -1.0
        for r in filtered[:8]:
            text = f"{r.get('title','')}. {r.get('snippet','')}"
            eprob = await nli_entails(text, claim)
            if numeric_consistent(claim, text): eprob = max(eprob, min(1.0, eprob + 0.1))
            if eprob > best_score: best_score = eprob; best = r
        if best:
            evidence.append({"claim": claim, "url": best.get("url",""), "title": best.get("title",""), "score": round(best_score,3)})
            if best_score >= 0.6: covered += 1

    coverage = covered / max(1, len(claims))
    verified = (coverage >= MIN_COVERAGE) and (len({e['url'] for e in evidence}) >= MIN_SOURCES)
    reason = f"coverage={coverage:.2f}, sources={len({e['url'] for e in evidence})}"
    return {"verified": bool(verified), "score": float(coverage), "reason": reason, "evidence": evidence[:10]}
