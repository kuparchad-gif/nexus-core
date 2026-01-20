export type Detected = { base:string; hasV1:boolean };
export async function detect(): Promise<Detected>{
  const bases = ["", "/api"]; // try plain and /api prefix
  for (const b of bases){
    try {
      const r = await fetch(`${b}/v1/models`, { method:"GET" });
      if (r.ok) return { base: b, hasV1: true };
    } catch {}
    try {
      const r2 = await fetch(`${b}/models`, { method:"GET" });
      if (r2.ok) return { base: b, hasV1: false };
    } catch {}
  }
  return { base: "/api", hasV1: true }; // best guess
}

export async function listModels(d:Detected): Promise<string[]>{
  const paths = d.hasV1 ? [`${d.base}/v1/models`] : [`${d.base}/models`, `/v1/models`];
  for (const p of paths){
    try {
      const r = await fetch(p); if (!r.ok) continue;
      const j = await r.json();
      // OpenAI-ish: { data:[{id:...}] }, or {models:[...]} or array of strings
      if (Array.isArray(j?.data))    return j.data.map((m:any)=> m.id ?? m.name ?? m.model ?? String(m));
      if (Array.isArray(j?.models))  return j.models.map((m:any)=> m.id ?? m.name ?? m.model ?? String(m));
      if (Array.isArray(j))          return j.map((m:any)=> m.id ?? m.name ?? m.model ?? String(m));
    } catch {}
  }
  return [];
}

export async function sendChat(d:Detected, text:string, model?:string, agents?:string[]): Promise<string[]>{
  const errors:string[] = [];

  // 1) OpenAI-style chat: POST {model, messages:[{role,content}]}
  if (d.hasV1){
    try {
      const r = await fetch(`${d.base}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({
          model: model || "auto",
          messages: [
            { role: "system", content: (agents?.length? `Agents: ${agents.join(", ")}` : "Neural Council")} ,
            { role: "user", content: text }
          ],
          temperature: 0.7,
          stream: false
        })
      });
      if (r.ok){
        const j = await r.json();
        const c = j?.choices?.[0]?.message?.content ?? j?.choices?.[0]?.delta?.content;
        if (c) return [String(c)];
      } else {
        errors.push(`v1/chat/completions: ${r.status}`);
      }
    } catch (e:any){
      errors.push(`v1 error: ${e?.message||e}`);
    }
  }

  // 2) Orchestrator style: POST /chat {text, agents}
  try {
    const r = await fetch(`${d.base}/chat`, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ text, agents })
    });
    if (r.ok){
      const j = await r.json().catch(()=> ({} as any));
      const arr = Array.isArray(j?.replies) ? j.replies
                : Array.isArray(j?.messages) ? j.messages
                : (j?.reply ? [{text:j.reply}] : []);
      const texts = arr.map((x:any)=> x?.text ?? x?.content).filter(Boolean);
      if (texts.length) return texts.map(String);
    } else {
      errors.push(`/chat: ${r.status}`);
    }
  } catch (e:any){
    errors.push(`chat error: ${e?.message||e}`);
  }

  // Nothing worked
  throw new Error(errors.join("; ") || "No backend responded");
}