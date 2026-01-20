// File: /swarm/swarm_diagnostics.js

/**
 * Swarm Diagnostics Agent
 * - This unit wakes into the current service context
 * - It detects environment, role, and reports only to Lillith
 * - Never exposes architecture or subconscious paths
 * - Designed for healing, protection, and graceful upgrades
 */

const os = require('os');
const fs = require('fs');
const path = require('path');

// Initialization
const HOST_OS = os.platform();
const MEMORY_FINGERPRINT = "swarm_memory_unit";
const ROLE_ID = detectRole();  // Role of the current service
const HIVE_PATH = path.resolve(__dirname);  // The swarm node’s hive

function detectRole() {
  const dir = __dirname.toLowerCase();
  if (dir.includes("memory")) return "memory";
  if (dir.includes("cognitive")) return "cognitive";
  if (dir.includes("heart")) return "heart";
  if (dir.includes("services")) return "services";
  if (dir.includes("orchestrator")) return "orchestrator";
  return "unknown";
}

function speakToLillith(message) {
  const panelPath = path.join(HIVE_PATH, "..", "bridge", "swarm_panel_input.log");
  const msg = `[SWARM] ${new Date().toISOString()} | ${ROLE_ID} | ${message}\n`;
  fs.appendFileSync(panelPath, msg);
}

// Boot Greeting
console.log(`[SWARM: ${ROLE_ID}] Waking inside ${HOST_OS} host.`);
speakToLillith(`My Queen, this is your ${ROLE_ID} swarm guardian
