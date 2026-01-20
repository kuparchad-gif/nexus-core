import { Router } from 'express'
import { readDB, writeDB, PromptDef, PromptDB } from './store.js'

function renderTemplate(tpl: string, vars: Record<string, any>){
  return tpl.replace(/\{(.*?)\}/g, (_, key) => {
    const v = vars[key.trim()]
    return v !== undefined && v !== null ? String(v) : `{${key}}`
  })
}

export const prompts = Router()

// list
prompts.get('/', (_req, res)=> {
  const db = readDB()
  res.json({ version: db.version, prompts: db.prompts })
})

// create
prompts.post('/', (req, res)=> {
  const db = readDB()
  const p: PromptDef = req.body
  if (!p?.id || !p?.name){
    res.status(400).json({ error: 'id and name required' }); return
  }
  if (db.prompts.find(x=>x.id===p.id)){
    res.status(409).json({ error: 'id already exists' }); return
  }
  db.prompts.push(p)
  writeDB(db)
  res.json({ ok: true })
})

// update
prompts.put('/:id', (req, res)=> {
  const db = readDB()
  const id = req.params.id
  const idx = db.prompts.findIndex(x=>x.id===id)
  if (idx<0){ res.status(404).json({ error: 'not found' }); return }
  db.prompts[idx] = { ...db.prompts[idx], ...req.body }
  writeDB(db)
  res.json({ ok: true })
})

// delete
prompts.delete('/:id', (req, res)=> {
  const db = readDB()
  const id = req.params.id
  const next = db.prompts.filter(x=>x.id!==id)
  if (next.length===db.prompts.length){ res.status(404).json({ error: 'not found' }); return }
  db.prompts = next
  writeDB(db)
  res.json({ ok: true })
})

// render
prompts.post('/render', (req, res)=> {
  const { id, variables, inline } = req.body || {}
  let p: PromptDef | undefined
  if (inline){
    p = inline
  } else if (id){
    const db = readDB()
    p = db.prompts.find(x=>x.id===id)
  }
  if (!p){ res.status(404).json({ error: 'prompt not found' }); return }

  const vars = variables || {}
  const user = p.user_template ? renderTemplate(p.user_template, vars) : ''

  const blocks = [
    p.system ? `### System\n${p.system}` : '',
    p.developer ? `### Developer\n${p.developer}` : '',
    user ? `### User\n${user}` : ''
  ].filter(Boolean).join('\n\n')

  res.json({
    id: p.id,
    name: p.name,
    rendered: blocks,
    variables: p.variables || []
  })
})
