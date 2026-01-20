import fs from 'fs';
import https from 'https';
import { WebSocket } from 'ws';
import { encode, decode } from './nim-js/codec.js';
import { seal, open } from './secure.js';

const PORT = parseInt(process.env.NIM_WSS_PORT || '8443', 10);
const HOST = process.env.NIM_WSS_HOST || 'localhost';
const CA   = process.env.NIM_TLS_CA || '';         // optional server CA pin
const CERT = process.env.NIM_TLS_CLIENT_CERT || ''; // optional for mTLS
const KEY  = process.env.NIM_TLS_CLIENT_KEY  || ''; // optional for mTLS
const TILES = parseInt(process.env.NIM_TILES || '48', 10);
const AEAD_KEY_HEX = process.env.NIM_AEAD_KEY || '';
const AEAD_KEY = AEAD_KEY_HEX ? Buffer.from(AEAD_KEY_HEX, 'hex') : null;

const url = `wss://${HOST}:${PORT}`;
const tls = {
  rejectUnauthorized: !!CA,
  ca: CA ? fs.readFileSync(CA) : undefined,
  cert: CERT ? fs.readFileSync(CERT) : undefined,
  key: KEY ? fs.readFileSync(KEY) : undefined,
  minVersion: 'TLSv1.3'
};

const ws = new WebSocket(url, { rejectUnauthorized: !!CA, ca: tls.ca, cert: tls.cert, key: tls.key });

ws.on('open', () => {
  const req = Buffer.from(JSON.stringify({prompt: process.argv[2] || "Say hi in five words"}), 'utf8');
  let payload = req;
  if (AEAD_KEY) payload = seal(AEAD_KEY, payload);
  const wire = encode(payload, TILES);
  ws.send(wire, { binary: true });
});

ws.on('message', (data, isBinary) => {
  const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
  let payload = decode(buf, TILES);
  if (AEAD_KEY) payload = open(AEAD_KEY, payload);
  const obj = JSON.parse(Buffer.from(payload).toString('utf8'));
  console.log(obj);
  ws.close();
});

ws.on('error', (err) => console.error('WS error:', err));
