import { request } from 'undici';

const BASE = process.env.LMSTUDIO_BASE_URL || 'http://127.0.0.1:1234';
let cachedModel: string | null = null;

export async function listModels() {
  const url = `${BASE}/v1/models`;
  const r = await request(url, { method: 'GET' });
  if (r.statusCode !== 200) throw new Error(`LM Studio /v1/models ${r.statusCode}`);
  return await r.body.json();
}

export async function chooseModel() {
  const pref = process.env.LMSTUDIO_MODEL || 'auto';
  const data = await listModels();
  const models: string[] = (data?.data || []).map((m: any) => m.id);

  if (pref !== 'auto') {
    const exists = models.includes(pref);
    cachedModel = exists ? pref : null;
  }

  if (!cachedModel) {
    // simple preference: pick first chat-capable or coder-ish model name
    const order = [/deepseek|codestral|qwen|llama|mistral/i, /coder|code|instruct/i];
    let picked = models[0];
    for (const rx of order) {
      const m = models.find(id => rx.test(id));
      if (m) { picked = m; break; }
    }
    cachedModel = picked || models[0];
  }

  return { models, selected: cachedModel };
}

export async function chat(opts: { system: string; user: string }) {
  const { selected } = await chooseModel();
  const url = `${BASE}/v1/chat/completions`;
  const r = await request(url, {
    method: 'POST',
    body: JSON.stringify({
      model: selected,
      messages: [
        { role: 'system', content: opts.system },
        { role: 'user', content: opts.user }
      ],
      temperature: 0.2,
      max_tokens: 256
    }),
    headers: { 'Content-Type': 'application/json' }
  });
  if (r.statusCode !== 200) {
    const text = await r.body.text();
    throw new Error(`chat failed ${r.statusCode}: ${text}`);
  }
  const data = await r.body.json();
  const content = data?.choices?.[0]?.message?.content ?? '';
  return { model: selected!, response: content };
}
