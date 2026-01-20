export type InferResp = { encoded_b64: string; tiles: number };
export async function infer(url: string, prompt: string, tiles = 64): Promise<InferResp> {
  const r = await fetch(`${url}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, stream: false, tiles }),
  });
  if (!r.ok) throw new Error(`infer failed: ${r.status}`);
  return r.json();
}

export async function nimWebSocket(url: string, encoded_b64: string, tiles = 64): Promise<any> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url.replace(/^http/, "ws") + "/nim/stream");
    ws.onopen = () => ws.send(JSON.stringify({ encoded_b64, tiles }));
    ws.onmessage = (ev) => { try { resolve(JSON.parse(ev.data)); } catch { resolve({ raw: ev.data }); } ws.close(); };
    ws.onerror = (e) => reject(e);
  });
}
