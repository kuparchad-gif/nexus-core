# bench/bench.py — run a prompt set against an OpenAI-compatible endpoint
import json, time, argparse, httpx, statistics, pathlib

def run(url:str, model:str, prompts_path:str, out_path:str):
    rows = []
    with open(prompts_path,"r",encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    with httpx.Client(timeout=120.0) as cli:
        for i, item in enumerate(lines, 1):
            body = {
                "model": model,
                "messages": item["messages"],
                "max_tokens": 256,
                "temperature": 0.2,
                "stream": False
            }
            t0 = time.time()
            r = cli.post(f"{url.rstrip('/')}/v1/chat/completions", json=body)
            dt = time.time() - t0
            try:
                data = r.json()
            except Exception:
                data = {"error": r.text}
            usage = data.get("usage") or {}
            toks = (usage.get("prompt_tokens",0) + usage.get("completion_tokens",0)) or 256
            rows.append({"i":i,"ms":int(dt*1000),"toks":toks,"toks_per_sec": toks/max(0.001,dt)})
    pathlib.Path(out_path).write_text(json.dumps({"rows":rows}, indent=2))
    tps = [r["toks_per_sec"] for r in rows if r.get("toks_per_sec")]
    print(f"Ran {len(rows)} prompts. p50={statistics.median(tps):.2f} tok/s, avg={sum(tps)/len(tps):.2f} tok/s.")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL (e.g., http://localhost:8007 for uce-router)")
    ap.add_argument("--model", default="/models/14b.gguf")
    ap.add_argument("--prompts", default="bench/prompts.jsonl")
    ap.add_argument("--out", default="bench/baseline.json")
    args = ap.parse_args()
    run(args.url, args.model, args.prompts, args.out)
