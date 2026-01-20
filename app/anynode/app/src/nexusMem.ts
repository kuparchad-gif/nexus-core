import fetch from "node-fetch";

export interface RememberReq {
  role?: string;
  text: string;
  tags?: string[];
  priority?: number;
  ttl_seconds?: number;
}

export class NexusMemClient {
  base: string;
  timeoutMs: number;

  constructor(baseUrl: string = "[REDACTED-URL] timeoutMs: number = 2000) {
    this.base = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
  }

[REDACTED-SECRET-LINE]
    const u = new URL(this.base + "/context");
[REDACTED-SECRET-LINE]
    if (topics && topics.length) u.searchParams.set("topics", topics.join(","));
    const r = await fetch(u.toString(), {method:"GET"});
    if (!r.ok) throw new Error(`context: ${r.status}`);
    return r.json();
  }

  async remember(req: RememberReq): Promise<any> {
    const r = await fetch(this.base + "/remember", {
      method:"POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({role:"system", priority:5, ...req})
    });
    if (!r.ok) throw new Error(`remember: ${r.status}`);
    return r.json();
  }

  async mirror(text: string, tags: string[] = [], priority = 8, vector?: number[]): Promise<any> {
    const r = await fetch(this.base + "/mirror", {
      method:"POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, tags, priority, vector })
    });
    if (!r.ok) throw new Error(`mirror: ${r.status}`);
    return r.json();
  }

  async health(): Promise<any> {
    const r = await fetch(this.base + "/healthz");
    if (!r.ok) throw new Error(`health: ${r.status}`);
    return r.json();
  }
}

