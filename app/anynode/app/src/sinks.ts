import fs from 'fs';
import { WebSocket } from 'ws';

const sinks = (process.env.SINKS || 'jsonl').split(',').map(s => s.trim());
const EVENT_LOG = process.env.EVENT_LOG || './data/psych_events.jsonl';
const WS_URL = process.env.WS_URL;

let ws: WebSocket | null = null;
if (sinks.includes('ws') && WS_URL) {
  ws = new WebSocket(WS_URL);
  ws.on('open', () => console.log('[sinks] WS connected'));
  ws.on('error', (e) => console.warn('[sinks] WS error', e.message));
}

export async function writeEvent(ev: any) {
  if (sinks.includes('jsonl')) {
    await fs.promises.mkdir(require('path').dirname(EVENT_LOG), { recursive: true });
    await fs.promises.appendFile(EVENT_LOG, JSON.stringify(ev) + '\n', 'utf8');
  }
  if (sinks.includes('ws') && ws && ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify({ type: 'psych_event', data: ev }));
  }
}
