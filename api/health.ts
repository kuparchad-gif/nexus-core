// Nexus Health Check — Vercel Backend Node

import type { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(_req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Content-Type", "application/json");

  return res.json({
    status: "healthy",
    platform: "vercel",
    role: "backend",
    timestamp: new Date().toISOString(),
  });
}
