#!/usr/bin/env python3
from __future__ import annotations
import os, re, sys, json, argparse, asyncio
from typing import List, Dict

def _optional_import(name: str):
    try: return __import__(name)
    except Exception: return None
httpx=_optional_import("httpx"); bs4=_optional_import("bs4"); readability=_optional_import("readability"); ddgs=_optional_import("ddgs")

import metatron_validator as mv
USER_AGENT="Mozilla/5.0 (compatible; NexusScraper/1.0)"

def extract_text(html_text: str) -> str:
    html_text = html_text or ""
    if readability is not None:
        try:
            doc = readability.Document(html_text)
            import re as _re
            return _re.sub(r"<[^>]+>", " ", doc.summary() or "")
        except Exception:
            pass
    if bs4 is not None:
        try:
            import bs4 as _bs4
            soup = _bs4.BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script","style","noscript","header","footer","nav","form"]):
                tag.decompose()
            return soup.get_text(" ", strip=True)
        except Exception:
            pass
    import re as _re
    return _re.sub(r"<[^>]+>", " ", html_text)

async def search(query: str, max_results: int = 6) -> List[Dict]:
    out = []
    if ddgs is not None:
        try:
            from ddgs import DDGS
            with DDGS() as d:
                for r in d.text(query, max_results=max_results or 6, region="us-en", safesearch="moderate"):
                    out.append({"title": r.get("title",""), "url": r.get("href") or r.get("url") or "", "snippet": r.get("body","")})
        except Exception:
            pass
    seen=set(); uniq=[]
    for r in out:
        u=r.get("url","")
        if u and u not in seen:
            seen.add(u); uniq.append(r)
    return uniq[:max_results or 6]

async def fetch(url: str, timeout: float = 8.0) -> str:
    if httpx is None:
        return ""
    try:
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
            raw = r.text
            return extract_text(raw)
    except Exception:
        return ""

def extract_questions(text: str, limit: int = 12) -> List[str]:
    cands = re.findall(r"([A-Z0-9][^?]{6,238}\?)", text, flags=re.M)
    seen=set(); out=[]
    for q in cands:
        k=re.sub(r"\W+","",q.lower())
        if k not in seen:
            seen.add(k); out.append(q.strip())
        if len(out)>=limit: break
    return out

async def scrape(query: str | None, urls: List[str], max_results: int, validate: bool) -> Dict:
    targets = []
    for u in (urls or []):
        targets.append({"url": u, "title": "", "snippet": ""})
    if query:
        for r in await search(query, max_results=max_results):
            targets.append(r)

    seen=set(); uniq=[]
    for t in targets:
        u=t.get("url","")
        if u and u not in seen:
            seen.add(u); uniq.append(t)

    items = []
    for t in uniq:
        txt = await fetch(t["url"])
        qs = extract_questions(txt, limit=12)
        for q in qs:
            item = {"question": q, "url": t["url"], "title": t.get("title",""), "snippet": t.get("snippet","")}
            if validate:
                res = await mv.validate(q, [t["url"]], max_results=4)
                item["validation"] = res
            items.append(item)

    return {"count": len(items), "items": items}

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scrape questions from the web and optionally validate them.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", help="Search query to find pages")
    g.add_argument("--urls", help="Comma-separated URLs")
    p.add_argument("--max-results", type=int, default=6)
    p.add_argument("--validate", action="store_true", help="Run validator on each found question")
    p.add_argument("--save-json", help="Save output JSON to this path")
    return p.parse_args(argv)

def _main(argv=None):
    args = _parse_args(argv)
    urls = [u.strip() for u in (args.urls.split(",") if args.urls else []) if u.strip()]
    async def _run():
        out = await scrape(args.query, urls, args.max_results, args.validate)
        js = json.dumps(out, ensure_ascii=False, indent=2)
        print(js)
        if args.save_json:
            open(args.save_json,"w",encoding="utf-8").write(js)
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    asyncio.run(_run())

if __name__=="__main__":
    _main()
