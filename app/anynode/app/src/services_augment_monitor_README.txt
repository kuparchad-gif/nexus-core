This patch expects `monitor` to expose a Loki proxy at POST /logs/query.

Minimal FastAPI handler to add to services/monitor/app.py:

    import os, httpx
    from fastapi import Body

    LOKI_URL = os.getenv("LOKI_URL","http://loki:3100")

    @app.post("/logs/query")
    async def logs_query(spec: dict = Body(...)):
        query = spec.get("query",'{}')
        start = spec.get("start")
        end = spec.get("end")
        limit = int(spec.get("limit", 1000))
        params = {"query": query, "limit": str(limit)}
        if start: params["start"] = str(start)
        if end: params["end"] = str(end)
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params)
            r.raise_for_status()
            raw = r.json()
        # normalize streams
        streams = []
        for st in raw.get("data",{}).get("result",[]):
            streams.append({"labels": st.get("stream",{}), "values": st.get("values",[])})
        return {"streams": streams}
