import express, { Request, Response } from 'express'

const router = express.Router()

type Msg = { role: 'system'|'user'|'assistant', content: string }

function ok(res: Response, data: any){ res.status(200).json(data) }
function bad(res: Response, code: number, msg: string){ res.status(code).json({ error: msg }) }

const LMSTUDIO = process.env.LMSTUDIO_TARGET || 'http://host.docker.internal:1234'
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL
const OPENAI_API_KEY = process.env.OPENAI_API_KEY

router.get('/providers', (_req, res)=>{
  const providers: string[] = []
  if (LMSTUDIO) providers.push('lmstudio')
  if (OPENAI_BASE_URL && OPENAI_API_KEY) providers.push('openai')
  ok(res, { providers })
})

router.get('/models', async (req, res)=>{
  try{
    const provider = String(req.query.provider||'lmstudio')
    if(provider==='lmstudio'){
      const r = await fetch(`${LMSTUDIO}/v1/models`, { headers:{ 'accept':'application/json' }})
      if(!r.ok) return bad(res, r.status, 'lmstudio models failed')
      const j = await r.json()
      const ids = Array.isArray(j?.data) ? j.data.map((m:any)=> m.id) : []
      return ok(res, { provider, models: ids })
    }
    if(provider==='openai'){
      const r = await fetch(`${OPENAI_BASE_URL}/v1/models`, { headers:{ 'Authorization': `Bearer ${OPENAI_API_KEY}` }})
      if(!r.ok) return bad(res, r.status, 'openai models failed')
      const j = await r.json()
      const ids = Array.isArray(j?.data) ? j.data.map((m:any)=> m.id) : []
      return ok(res, { provider, models: ids })
    }
    return bad(res, 400, 'unknown provider')
  }catch(e:any){
    return bad(res, 500, e?.message||'models error')
  }
})

router.post('/chat', express.json({ limit:'2mb' }), async (req, res)=>{
  try{
    const { provider='lmstudio', model, messages } = req.body || {}
    if(!Array.isArray(messages) || !model) return bad(res, 400, 'model and messages[] required')

    if(provider==='lmstudio'){
      const r = await fetch(`${LMSTUDIO}/v1/chat/completions`, {
        method:'POST',
        headers:{ 'content-type': 'application/json' },
        body: JSON.stringify({ model, messages, stream:false }),
      })
      const txt = await r.text()
      if(!r.ok) return bad(res, r.status, txt || 'lmstudio chat failed')
      const j = JSON.parse(txt)
      const content = j?.choices?.[0]?.message?.content ?? ''
      return ok(res, { content, raw: j })
    }

    if(provider==='openai'){
      if(!OPENAI_BASE_URL || !OPENAI_API_KEY) return bad(res, 400, 'openai env not set')
      const r = await fetch(`${OPENAI_BASE_URL}/v1/chat/completions`, {
        method:'POST',
        headers:{ 'content-type': 'application/json', 'authorization': `Bearer ${OPENAI_API_KEY}` },
        body: JSON.stringify({ model, messages, stream:false }),
      })
      const txt = await r.text()
      if(!r.ok) return bad(res, r.status, txt || 'openai chat failed')
      const j = JSON.parse(txt)
      const content = j?.choices?.[0]?.message?.content ?? ''
      return ok(res, { content, raw: j })
    }

    return bad(res, 400, 'unknown provider')
  }catch(e:any){
    return bad(res, 500, e?.message||'chat error')
  }
})

export default router
