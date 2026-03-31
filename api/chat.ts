// Nexus Chat Relay — Vercel Backend Node
// Routes chat requests to Workers mesh, falls back to Modal quantum layer

import type { VercelRequest, VercelResponse } from "@vercel/node";

const WORKERS = Array.from(
  { length: 80 },
  (_, i) =>
    `https://nexus-universal-${String(i + 1).padStart(3, "0")}.kuparchad.workers.dev`
);
const rw = () => WORKERS[Math.floor(Math.random() * WORKERS.length)];

const MODAL_URL =
  process.env.MODAL_URL ||
  "https://aethereal-nexus-viren-db0--sovereign-edge-sovereign-nexu-b7f1c3.modal.run";

const cors = {
  "Access-Control-Allow-Origin": "*",
  "Content-Type": "application/json",
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  Object.entries(cors).forEach(([k, v]) => res.setHeader(k, v));

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const body = req.body || {};
  const worker = rw();

  // Try Workers mesh first
  try {
    const r = await fetch(`${worker}/api/v1/nexus-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (r.ok) {
      const data = await r.json();
      return res.json({ ...data, relay: "vercel", routed_to: worker });
    }
  } catch {
    // Worker failed, fall through to Modal
  }

  // Fallback: Modal quantum layer
  try {
    const r = await fetch(`${MODAL_URL}/api/v1/nexus-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    return res.json({ ...data, relay: "vercel", routed_to: "modal" });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : "unknown error";
    return res.status(502).json({ error: msg, relay: "vercel" });
  }
}
