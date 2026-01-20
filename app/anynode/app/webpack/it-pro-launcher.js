// Lillith Launch Script (Web + API + Chat UI + Zen Mode)
// Auto-starts server and opens web portal in default browser with calming visual toggle

const { exec } = require('child_process');
const path = require('path');

let open;
try {
  open = require('open');
} catch (err) {
  // Fallback to dynamic import if using ESM version of open
  import('open').then(mod => {
    open = mod.default;
    open(`http://localhost:${PORT}`);
  }).catch(err => {
    console.warn('[FAIL] Could not dynamically load `open`:', err.message);
    console.log(`[MANUAL] Open browser to http://localhost:${PORT}`);
  });
}

const SERVER_PATH = path.join(__dirname, 'lillith-tenticals-professionalv25.js');
const PORT = 5003;

console.log('[BOOT] Launching Lillith Queen Professional backend...');
const backend = exec(`node "${SERVER_PATH}"`, (err, stdout, stderr) => {
  if (err) {
    console.error('[ERROR] Failed to launch backend:', err);
    return;
  }
  if (stdout) console.log(stdout);
  if (stderr) console.error(stderr);
});

setTimeout(() => {
  const url = `http://localhost:${PORT}`;
  console.log(`[BOOT] Opening browser at ${url}`);
  if (typeof open === 'function') {
    open(url).catch(err => console.error('[ERROR] Failed to open browser:', err));
  }
}, 3000);

backend.stdout?.on('data', data => process.stdout.write(`[LILLITH]: ${data}`));
backend.stderr?.on('data', data => process.stderr.write(`[LILLITH ERROR]: ${data}`));
