#!/usr/bin/env node
const REGISTRY_ENDPOINT = process.env.REGISTRY_URL || 'https://nexus-universal.your-account.workers.dev';

async function discoverAllNetworks(keyword = 'universal') {
  console.log(`🔍 Discovering workers matching "${keyword}"...`);
  const res = await fetch(`${REGISTRY_ENDPOINT}/registry/discover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keyword })
  });
  const data = await res.json();
  console.log(`\n📊 Found ${data.found || 0} workers:`);
  for (const w of data.workers || []) {
    console.log(`  • ${w.name || w.id} (${w.networks?.join(', ') || 'unknown'})`);
  }
  return data;
}

const [,, cmd, ...args] = process.argv;

if (cmd === 'discover') {
  discoverAllNetworks(args[0]).then(console.log);
} else if (cmd === 'register') {
  const [id, url, networks = 'pulse,pubhub,legacy'] = args;
  if (!id || !url) {
    console.error('Usage: node network-discover.js register <id> <url> [networks]');
    process.exit(1);
  }
  fetch(`${REGISTRY_ENDPOINT}/registry/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      workerId: id,
      name: `hypercore-${id}`,
      endpoint: url,
      networks: networks.split(','),
      preferredNetwork: networks.split(',')[0]
    })
  }).then(r => r.json()).then(console.log);
} else {
  console.log(`
Usage:
  node network-discover.js discover [keyword]     # Discover workers
  node network-discover.js register <id> <url> [networks]  # Register worker
`);
}
