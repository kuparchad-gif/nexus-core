#!/usr/bin/env python3
# (code abbreviated here for brevity in the notebook; the full content will be written)
from __future__ import annotations
import os, re, sys, json, math, html, argparse, asyncio, hashlib
from typing import List, Dict, Optional, Tuple
# Optional imports
def _optional_import(name: str):
    try: return __import__(name)
    except Exception: return None
httpx=_optional_import("httpx"); bs4=_optional_import("bs4"); readability=_optional_import("readability"); ddgs=_optional_import("ddgs"); transformers=_optional_import("transformers")
MIN_SOURCES=int(os.getenv("VERIFY_MIN_SOURCES","2")); MIN_COVERAGE=float(os.getenv("VERIFY_MIN_COVERAGE","0.7")); MAX_AGE_DAYS=int(os.getenv("VERIFY_MAX_AGE_DAYS","365"))
MNLI_MODEL_NAME=os.getenv("MNLI_MODEL_NAME","roberta-large-mnli"); USER_AGENT="Mozilla/5.0 (compatible; NexusValidator/1.0)"
import time, math as _m, html as _h
def sha1(s:str)->str: return hashlib.sha1(s.encode("utf-8","ignore")).hexdigest()
def now_iso():
    import datetime; return datetime.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"
def split_claims(text:str)->List[str]:
    import re
    parts=re.split(r"(?<=[.!?])\s+|(?:\s+and\s+|\s+but\s+)", text.strip(), flags=re.I)
    out=[]; seen=set()
    for p in parts:
        p=p.strip()
        if len(p)>2:
            k=re.sub(r"\W+","",p.lower())
            if k not in seen: seen.add(k); out.append(p)
    return out[:12]
import re as _re
SCRUB_EMAIL=_re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SCRUB_PHONE=_re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b")
SCRUB_HANDLE=_re.compile(r"(?<!\w)@([A-Za-z0-9_]{2,30})\b")
SCRUB_IP=_re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SCRUB_ADDR=_re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\- ]{3,}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b", _re.I)
def scrub_text(s:str)->str:
    s=SCRUB_EMAIL.sub("[redacted-email]",s)
    s=SCRUB_PHONE.sub("[redacted-phone]",s)
    s=SCRUB_HANDLE.sub("@[redacted]",s)
    s=SCRUB_IP.sub("[redacted-ip]",s)
    s=SCRUB_ADDR.sub("[redacted-address]",s)
    return _re.sub(r"\s+"," ",s).strip()[:2000]
def jaccard(a:str,b:str)->float:
    sa=set(_re.findall(r"[A-Za-z0-9]+",a.lower())); sb=set(_re.findall(r"[A-Za-z0-9]+",b.lower()))
    return (len(sa&sb)/len(sa|sb)) if sa and sb else 0.0
def sequence_overlap(a:str,b:str)->float:
    ta=_re.findall(r"\w+",a.lower()); tb=_re.findall(r"\w+",b.lower())
    from collections import Counter
    ca,cb=Counter(ta),Counter(tb); inter=sum(min(ca[t],cb[t]) for t in set(ca)|set(cb)); union=sum(max(ca[t],cb[t]) for t in set(ca)|set(cb))
    return inter/union if union else 0.0
def metatrons_cube_points(r:float=1.0):
    pts={"C0":(0.0,0.0)}
    for i in range(6):
        ang=_m.radians(60*i); pts[f"I{i}"]=(_m.cos(ang)*r,_m.sin(ang)*r)
    phi=(1+5**0.5)/2; R=phi*r
    for i in range(6):
        ang=_m.radians(60*i+30); pts[f"O{i}"]=(_m.cos(ang)*R,_m.sin(ang)*R)
    return pts
def metatron_apply(terms:List[str]):
    nodes=list(metatrons_cube_points().keys())
    terms=(terms+["∅"]*13)[:13]; return {n:t for n,t in zip(nodes,terms)}
async def search_web(query:str,max_results:int=6)->List[Dict]:
    results=[]
    if ddgs is not None:
        try:
            from ddgs import DDGS
            with DDGS() as d:
                for r in d.text(query,max_results=max_results or 6, region="us-en", safesearch="moderate"):
                    results.append({"title":r.get("title",""),"url":r.get("href") or r.get("url","") or "","snippet":r.get("body","")})
        except Exception: pass
    # other providers via env could be added here (omitted in compact version)
    seen=set(); out=[]
    for r in results:
        u=r.get("url","")
        if u and u not in seen: seen.add(u); out.append(r)
    return out[:max_results or 6]
def extract_text(html_text:str)->str:
    html_text=_h.unescape(html_text or "")
    if readability is not None:
        try:
            doc=readability.Document(html_text); import re as __re
            return scrub_text(__re.sub(r"<[^>]+>"," ",doc.summary() or ""))
        except Exception: pass
    if bs4 is not None:
        try:
            import bs4 as _bs4
            soup=_bs4.BeautifulSoup(html_text,"html.parser")
            for tag in soup(["script","style","noscript","header","footer","nav","form"]): tag.decompose()
            return scrub_text(soup.get_text(" ", strip=True))
        except Exception: pass
    import re as __re
    return scrub_text(__re.sub(r"<[^>]+>"," ",html_text))
async def fetch(url:str,timeout:float=8.0)->str:
    if httpx is None: return ""
    try:
        async with httpx.AsyncClient(headers={"User-Agent":USER_AGENT}, timeout=timeout, follow_redirects=True) as client:
            r=await client.get(url)
            ct=r.headers.get("content-type","")
            raw=r.text
            return extract_text(raw) if "html" in ct.lower() else raw
    except Exception: return ""
_nli=None
def get_nli():
    global _nli
    if _nli is not None: return _nli
    if transformers is None: return None
    try:
        from transformers import pipeline
        _nli=pipeline("zero-shot-classification", model=os.getenv("MNLI_MODEL_NAME","roberta-large-mnli"))
        return _nli
    except Exception: return None
def nli_entails(premise:str,hypothesis:str)->float:
    pipe=get_nli()
    if pipe is None: return max(jaccard(premise,hypothesis), sequence_overlap(premise,hypothesis))
    out=pipe(hypothesis, candidate_labels=["entailed","contradicted","unrelated"], hypothesis_template="{}")
    labels=[l.lower() for l in out["labels"]]; scores=out["scores"]; d=dict(zip(labels,scores))
    return float(d.get("entailed",0.0))
SUBJECTIVE_WORDS=set("believe think feel seems appears arguably probably maybe perhaps likely unlikely should could moral spiritual sacred metaphysical angelic demonic karmic destiny fate energy vibe aura".split())
SCIENCE_MARKERS=[r"\\b10\\.\\d{4,9}/[-._;()/:A-Z0-9]+\\b", r"\\barXiv:\\d{4}\\.\\d{4,5}\\b", r"\\b(pubmed|randomized|controlled trial|systematic review|meta-analysis|peer[- ]review)\\b"]
SPIRITUAL_MARKERS=r"\\b(spiritual|sacred geometry|metatron|kabbalah|chakra|meditation|esoteric|occult|alchemy|ritual|prayer)\\b"
METAPHOR_MARKERS=r"\\b(as if|like a|symbol(?:izes|ic)|metaphor|allegor|parable)\\b"
def flag_fact_opinion(claim:str, contexts:List[str]):
    pipe=get_nli()
    if pipe is not None:
        out=pipe(claim, candidate_labels=["fact","opinion"], multi_label=True)
        labels=[l.lower() for l in out["labels"]]; scores=out["scores"]; d=dict(zip(labels,scores))
        is_fact=d.get("fact",0.0)>=0.5; is_op=d.get("opinion",0.0)>=0.5 and (not is_fact or d.get("opinion",0.0)>d.get("fact",0.0)); conf=float(max(d.get("fact",0.0), d.get("opinion",0.0)))
        return is_fact,is_op,conf
    toks=_re.findall(r"\\w+", claim.lower()); subj=sum(1 for t in toks if t in SUBJECTIVE_WORDS); is_op=subj>=2 or ("?" in claim)
    best=max((sequence_overlap(claim,c) for c in contexts), default=0.0); is_fact=(best>=0.45) and not is_op; conf=max(best, 0.5 if is_op else 0.6 if is_fact else 0.4)
    return is_fact,is_op,conf
def flag_scientific(contexts:List[str]):
    text=" ".join(contexts).lower(); import re as __re; hits=0
    for pat in SCIENCE_MARKERS:
        if __re.search(pat, text, flags=__re.I): hits+=1
    return (hits>=1, min(1.0, 0.4+0.3*hits))
def flag_spiritual(contexts:List[str]):
    import re as __re; text=" ".join(contexts); m=__re.search(SPIRITUAL_MARKERS, text, flags=__re.I); return (m is not None, 0.6 if m else 0.2)
def flag_metaphor(claim:str, contexts:List[str]):
    import re as __re; text=" ".join([claim]+contexts); m=__re.search(METAPHOR_MARKERS, text, flags=__re.I); is_meta=(m is not None)
    if not is_meta and __re.search(r"\\blike\\b", claim.lower()): is_meta=True
    return (is_meta, 0.5 if is_meta else 0.2)
class SourceDoc:
    def __init__(self,url:str,title:str="",snippet:str="",text:str=""):
        self.url=url; self.title=title; self.snippet=snippet; self.text=text
async def gather_sources(query:Optional[str], explicit_sources:List[str], max_results:int):
    res=[]
    for u in (explicit_sources or []): res.append(SourceDoc(url=u.strip()))
    if query:
        for r in (await search_web(query, max_results=max_results)) or []: res.append(SourceDoc(url=r.get("url",""), title=r.get("title",""), snippet=r.get("snippet","")))
    seen=set(); out=[]
    for s in res:
        if not s.url or s.url in seen: continue
        seen.add(s.url)
        if httpx is not None: s.text=await fetch(s.url)
        out.append(s)
    return out
def evidence_for_claim(claim:str, docs:List[SourceDoc]):
    ev=[]
    for d in docs:
        if not d.text: continue
        import re as __re
        sents=__re.split(r"(?<=[.!?])\\s+", d.text)
        best=None; best_score=-1.0
        for s in sents[:200]:
            score=0.6*sequence_overlap(claim,s)+0.4*jaccard(claim,s)
            if score>best_score: best_score=score; best=s
        if best_score>=0.35 and best: ev.append({"url":d.url,"title":d.title,"snippet":scrub_text(best),"score":round(best_score,3)})
    ev.sort(key=lambda x:x["score"], reverse=True); return ev[:8]
def topic_terms_from_claim(claim:str)->List[str]:
    import re as __re
    toks=__re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", claim)
    stop=set("the and or for with into from about this that those these like very more less good bad causes makes using usage use will may might could should".split())
    out=[]; seen=set()
    for t in toks:
        if t.lower() not in stop and t.lower() not in seen: seen.add(t.lower()); out.append(t)
    return out[:13]
async def validate(query:str, explicit_sources:List[str], max_results:int=6)->Dict:
    claims=split_claims(query); docs=await gather_sources(query, explicit_sources, max_results=max_results)
    contexts=[d.text[:1500] for d in docs if d.text]
    all_evidence={}; covered=0; distinct=set()
    for c in claims:
        ev=evidence_for_claim(c, docs); all_evidence[c]=ev
        if ev:
            entails=[nli_entails(e["snippet"], c) for e in ev[:2]]
            es=max(entails) if entails else 0.0
            if es>=0.6 or ev[0]["score"]>=0.5: covered+=1; distinct.add(ev[0]["url"])
    coverage=covered/max(1,len(claims))
    is_fact,is_op,fc=flag_fact_opinion(query, contexts)
    is_sci,sc=flag_scientific(contexts); has_spirit,spc=flag_spiritual(contexts); is_meta,mc=flag_metaphor(query, contexts)
    terms=topic_terms_from_claim(query); mapping=metatron_apply(terms)
    verified=(coverage>=float(os.getenv("VERIFY_MIN_COVERAGE","0.7"))) and (len(distinct)>=int(os.getenv("VERIFY_MIN_SOURCES","2")))
    reason=f"coverage={coverage:.2f}, sources={len(distinct)}"
    sources_out=[{"url":d.url,"title":scrub_text(d.title or ""),"snippet":scrub_text(d.snippet or "")} for d in docs]
    return {"query":query,"timestamp":now_iso(),"verified":bool(verified),"score":round(coverage,3),"reason":reason,
            "flags":{"is_fact":bool(is_fact),"is_opinion":bool(is_op),"is_scientifically_proven":bool(is_sci),"spiritual_applications":bool(has_spirit),"is_metaphor":bool(is_meta),
                     "confidences":{"fact_or_opinion":round(fc,3),"scientific":round(sc,3),"spiritual":round(spc,3),"metaphor":round(mc,3)}},
            "metatron":{"terms":terms,"node_mapping":mapping,"notes":"Center=essence; inner=facets; outer=context/constraints."},
            "sources":sources_out[:max_results],"evidence":all_evidence,
            "notice":"Snippets scrubbed; only publicly fetchable sources used."}
def parse_args(argv=None):
    import argparse
    p=argparse.ArgumentParser(description="Validate a claim from multiple sources + Metatron mapping")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query"); g.add_argument("--input-file")
    p.add_argument("--sources", nargs="*", default=[]); p.add_argument("--max-results", type=int, default=6)
    p.add_argument("--no-search", action="store_true"); p.add_argument("--save-json")
    return p.parse_args(argv)
def main(argv=None):
    args=parse_args(argv)
    query=(open(args.input_file,"r",encoding="utf-8").read().strip() if args.input_file else args.query.strip())
    q_for_search=None if args.no_search else query
    async def _run():
        out=await validate(q_for_search or query, args.sources, max_results=args.max_results)
        js=json.dumps(out, ensure_ascii=False, indent=2); print(js)
        if args.save_json: open(args.save_json,"w",encoding="utf-8").write(js)
    try:
        if sys.platform.startswith("win"):
            import asyncio; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception: pass
    import asyncio; asyncio.run(_run())
if __name__=="__main__": main()
