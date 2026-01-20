import { Router } from 'express'
import { randomUUID } from 'crypto'

type Check = {
  key: string
  label: string
  run: (params: any) => Promise<{ ok: boolean, details?: any }>
}

type Troubleshooter = {
  id: string
  name: string
  description: string
  targets?: string[]
  checks: Check[]
}

export const troubleBus = new Map<string, any>() // jobId -> { status, logs: string[], result?: any }
export const troubleWSClients: Set<any> = new Set()

// Simple emit to WS clients filtered by jobId
function emit(jobId: string, payload: any){
  const msg = JSON.stringify({ channel: 'troubleshooters', jobId, ...payload })
  for (const ws of troubleWSClients){
    try{ ws.send(msg) }catch{}
  }
}

// --- Mock checks (replace with real system probes) ---

const wait = (ms: number)=> new Promise(res=>setTimeout(res, ms))

const PortDoctor: Troubleshooter = {
  id: 'port-doctor', name: 'Port Doctor',
  description: 'Scans critical service ports and reports conflicts or listeners.',
  targets: ['gateway','lillith','orc'],
  checks: [
    { key: 'p26', label: 'Port 26', run: async(_)=>{ await wait(400); return { ok: Math.random() > 0.15, details: { port: 26 } } } },
    { key: 'p443', label: 'Port 443', run: async(_)=>{ await wait(400); return { ok: true, details: { port: 443, tls: true } } } },
    { key: 'p8080', label: 'Port 8080', run: async(_)=>{ await wait(400); return { ok: Math.random() > 0.3, details: { port: 8080 } } } },
    { key: 'p1313', label: 'Port 1313', run: async(_)=>{ await wait(400); return { ok: Math.random() > 0.2, details: { port: 1313 } } } },
  ]
}

const GPUDoctor: Troubleshooter = {
  id: 'gpu-doctor', name: 'GPU Doctor',
  description: 'Inspects GPU availability, VRAM headroom, and driver sanity.',
  targets: ['models','training'],
  checks: [
    { key: 'cuda', label: 'CUDA presence', run: async(_)=>{ await wait(500); return { ok: Math.random() > 0.1, details: { cuda: Math.random()>0.5 ? '12.2' : '—' } } } },
    { key: 'vram', label: 'VRAM capacity', run: async(_)=>{ await wait(500); return { ok: Math.random() > 0.2, details: { freeGB: +(Math.random()*8+2).toFixed(1) } } } },
    { key: 'driver', label: 'Driver health', run: async(_)=>{ await wait(500); return { ok: Math.random() > 0.1, details: { version: '555.xx' } } } },
  ]
}

const LatencySleuth: Troubleshooter = {
  id: 'latency-sleuth', name: 'Latency Sleuth',
  description: 'Measures round-trip latency across service hops (Orc → Lillith → Memory).',
  targets: ['routing'],
  checks: [
    { key: 'orc', label: 'Orc hop', run: async(_)=>{ await wait(350); return { ok: true, details: { ms: +(Math.random()*20+5).toFixed(1) } } } },
    { key: 'lillith', label: 'Lillith hop', run: async(_)=>{ await wait(350); return { ok: true, details: { ms: +(Math.random()*25+10).toFixed(1) } } } },
    { key: 'memory', label: 'Memory hop', run: async(_)=>{ await wait(350); return { ok: Math.random()>0.05, details: { ms: +(Math.random()*25+10).toFixed(1) } } } },
  ]
}

export const REGISTRY: Troubleshooter[] = [PortDoctor, GPUDoctor, LatencySleuth]

export const troubleshooters = Router()

troubleshooters.get('/', (_req, res)=> {
  res.json({ troubleshooters: REGISTRY.map(t=> ({
    id: t.id, name: t.name, description: t.description, targets: t.targets||[], checks: t.checks.map(c=>({key:c.key,label:c.label}))
  }))})
})

troubleshooters.post('/run', async (req, res)=> {
  const { id, params } = req.body || {}
  const t = REGISTRY.find(x=>x.id===id)
  if (!t){ res.status(404).json({ error: 'not found' }); return }
  const jobId = randomUUID()
  troubleBus.set(jobId, { status: 'running', logs: [] })
  res.json({ ok: true, jobId })

  ;(async()=>{
    emit(jobId, { type: 'start', name: t.name })
    const results: Record<string, any> = {}
    for (const check of t.checks){
      emit(jobId, { type: 'log', message: `Running: ${check.label}` })
      try{
        const r = await check.run(params)
        results[check.key] = r
        emit(jobId, { type: 'log', message: `${check.label}: ${r.ok?'OK':'ISSUE'}` })
      }catch(e:any){
        results[check.key] = { ok:false, error: String(e) }
        emit(jobId, { type: 'log', message: `${check.label}: ERROR` })
      }
    }
    const ok = Object.values(results).every((r:any)=> r.ok)
    troubleBus.set(jobId, { status: ok ? 'ok' : 'needs_attention', logs: [], result: results })
    emit(jobId, { type: 'done', ok, results })
  })()
})

troubleshooters.get('/jobs/:id', (req, res)=> {
  const s = troubleBus.get(req.params.id)
  if (!s){ res.status(404).json({ error: 'not found' }); return }
  res.json(s)
})
