import { Router } from 'express'
import { randomUUID } from 'crypto'

export const trainingWSClients: Set<any> = new Set()

type Step = { key: string, label: string, durationMs: number }
type Curriculum = {
  id: string
  name: string
  description: string
  steps: Step[]
}

const wait = (ms:number)=> new Promise(res=>setTimeout(res, ms))

export const CURRICULA: Curriculum[] = [
  {
    id: 'prompt-mastery',
    name: 'Prompt Mastery',
    description: 'Design, variant testing, and safety harnesses for prompt engineering.',
    steps: [
      { key: 'draft', label: 'Draft variants', durationMs: 800 },
      { key: 'eval', label: 'Evaluate outputs', durationMs: 1200 },
      { key: 'harness', label: 'Apply safety harness', durationMs: 900 },
      { key: 'finalize', label: 'Finalize pack', durationMs: 600 },
    ]
  },
  {
    id: 'ops-hardening',
    name: 'Ops Hardening',
    description: 'Ports, routing, resilience, circuit breakers, chaos drill.',
    steps: [
      { key: 'ports', label: 'Verify ports', durationMs: 700 },
      { key: 'routing', label: 'Route checks', durationMs: 900 },
      { key: 'resilience', label: 'Resilience tests', durationMs: 1100 },
      { key: 'chaos', label: 'Chaos drill', durationMs: 1000 },
    ]
  },
  {
    id: 'model-loading-lab',
    name: 'Model Loading Lab',
    description: 'VRAM budgeting, quantization checks, loader fallbacks.',
    steps: [
      { key: 'budget', label: 'VRAM budget calc', durationMs: 800 },
      { key: 'quant', label: 'Quantization matrix', durationMs: 1000 },
      { key: 'loader', label: 'Loader fallback test', durationMs: 900 },
      { key: 'report', label: 'Report & next actions', durationMs: 600 },
    ]
  }
]

export const training = Router()

training.get('/curricula', (_req, res)=> {
  res.json({ curricula: CURRICULA })
})

training.post('/run', async (req, res)=> {
  const { id } = req.body || {}
  const c = CURRICULA.find(x=>x.id===id)
  if (!c){ res.status(404).json({ error: 'not found' }); return }
  const jobId = randomUUID()
  res.json({ ok: true, jobId })
  ;(async()=>{
    const total = c.steps.reduce((a,b)=>a+b.durationMs, 0)
    let elapsed = 0
    for (const step of c.steps){
      trainingEmit({ jobId, type: 'step', status: 'start', key: step.key, label: step.label, progress: Math.round(100*elapsed/total) })
      await wait(step.durationMs)
      elapsed += step.durationMs
      trainingEmit({ jobId, type: 'step', status: 'end', key: step.key, label: step.label, progress: Math.round(100*elapsed/total) })
    }
    trainingEmit({ jobId, type: 'done', progress: 100 })
  })()
})

function trainingEmit(payload:any){
  const msg = JSON.stringify({ channel: 'training', ...payload })
  for (const ws of trainingWSClients){
    try{ ws.send(msg) }catch{}
  }
}
