import os, json, http.client, urllib.parse
class LLMRequestError(Exception): pass
def http_json(method, base, path, headers, body=None, timeout=60):
    url = urllib.parse.urlparse(base)
    conn = http.client.HTTPSConnection(url.netloc, timeout=timeout)
    conn.request(method, path, body, headers)
    resp = conn.getresponse(); data = resp.read()
    if resp.status >= 300: raise LLMRequestError(f"{resp.status} {resp.reason}: {data[:400]}")
    try: return json.loads(data.decode())
    except Exception: return {"raw": data.decode(errors="ignore")}
