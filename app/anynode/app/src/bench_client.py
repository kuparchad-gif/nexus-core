# bench_client.py — simple async benchmark against OpenAI-compatible /v1/chat/completions
import argparse, asyncio, aiohttp, time, statistics, random

PROMPTS = [
  "Explain in ~200 tokens why speculative decoding improves perceived latency on memory-bound LLMs.",
  "In ~200 tokens, outline how to shard a dense 180B model across 8 GPUs using tensor parallelism.",
  "In ~200 tokens, compare CPU-first vs GPU-first inference strategies for chat agents.",
]

async def one_call(session, base_url, prompt):
    t0 = time.perf_counter()
    async with session.post(f"{base_url}/v1/chat/completions", json={
        "messages": [{"role":"user","content": prompt}], "stream": True
    }) as resp:
        if resp.status != 200:
            text = await resp.text()
            return {"ok": False, "status": resp.status, "err": text}
        ttft = None
        tokens = 0
        async for chunk in resp.content.iter_chunked(1024):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            tokens += max(1, len(chunk)//4)
        total_ms = (time.perf_counter() - t0) * 1000
        tps = (tokens / (total_ms/1000)) if total_ms>0 else 0
        return {"ok": True, "ttft_ms": ttft, "total_ms": total_ms, "tokens": tokens, "tps": tps}

async def run(base_url, concurrency, rounds):
    stats = []
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for r in range(rounds):
            prompts = random.sample(PROMPTS, k=min(concurrency, len(PROMPTS)))
            tasks = [one_call(session, base_url, p) for p in prompts]
            results = await asyncio.gather(*tasks)
            ok = [x for x in results if x.get("ok")]
            if not ok:
                print("Round", r+1, "failed:", results); continue
            ttft = [x["ttft_ms"] for x in ok]
            total = [x["total_ms"] for x in ok]
            tps = [x["tps"] for x in ok]
            import statistics
            print(f"Round {r+1}: TTFT p50={statistics.median(ttft):.0f}ms | Total p50={statistics.median(total):.0f}ms | TPS p50={statistics.median(tps):.1f}")
            stats.extend(ok)
    if stats:
        ttft = [x["ttft_ms"] for x in stats]
        total = [x["total_ms"] for x in stats]
        tps = [x["tps"] for x in stats]
        print("== Aggregate ==")
        print(f"TTFT p50={statistics.median(ttft):.0f}ms")
        print(f"Total p50={statistics.median(total):.0f}ms")
        print(f"TPS p50={statistics.median(tps):.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(run(args.base_url, args.concurrency, args.rounds))
