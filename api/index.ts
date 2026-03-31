// Nexus Backend Relay — Vercel Serverless Functions
// Routes: /api → status, /api/health → health check
// Part of the Nexus mesh — Vercel backend node

import type { VercelRequest, VercelResponse } from "@vercel/node";

const WORKERS = Array.from(
  { length: 80 },
  (_, i) =>
    `https://nexus-universal-${String(i + 1).padStart(3, "0")}.kuparchad.workers.dev`
);
const rw = () => WORKERS[Math.floor(Math.random() * WORKERS.length)];

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Content-Type", "application/json");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  return res.json({
    name: "Nexus Backend Relay",
    platform: "vercel",
    version: "1.0.0",
    role: "backend",
    workers: WORKERS.length,
    sample_worker: rw(),
    endpoints: ["/api/chat", "/api/task", "/api/dashboard", "/api/ask", "/api/health"],
  });
}
