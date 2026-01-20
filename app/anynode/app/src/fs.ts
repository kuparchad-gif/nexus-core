import express, { Request, Response } from 'express'
import fs from 'fs'
import fsp from 'fs/promises'
import path from 'path'
import multer from 'multer'

const router = express.Router()

const ROOT = process.env.FS_ROOT || '/workspace'
const upload = multer({ storage: multer.memoryStorage(), limits:{ fileSize: 50 * 1024 * 1024 } }) // 50MB

function norm(p: string){
  const safe = path.normalize(path.isAbsolute(p) ? p : path.join(ROOT, p))
  if(!safe.startsWith(path.normalize(ROOT))) throw new Error('Path outside FS_ROOT')
  return safe
}

router.get('/status', (_req, res)=> res.json({ root: ROOT }))

router.get('/list', async (req, res)=>{
  try{
    const p = String(req.query.path||ROOT)
    const abs = norm(p)
    const ents = await fsp.readdir(abs, { withFileTypes: true })
    const items = ents.map(e=>({ name:e.name, dir:e.isDirectory(), file:e.isFile() }))
    res.json({ path: abs, items })
  }catch(e:any){ res.status(500).json({ error: e?.message || 'list error' }) }
})

router.get('/read', async (req, res)=>{
  try{
    const abs = norm(String(req.query.path))
    const data = await fsp.readFile(abs, 'utf-8')
    res.type('text/plain').send(data)
  }catch(e:any){ res.status(500).json({ error: e?.message || 'read error' }) }
})

router.post('/write', express.json({ limit:'10mb' }), async (req, res)=>{
  try{
    const { path: p, content } = req.body || {}
    const abs = norm(String(p))
    await fsp.mkdir(path.dirname(abs), { recursive: true })
    await fsp.writeFile(abs, content ?? '', 'utf-8')
    res.json({ ok:true })
  }catch(e:any){ res.status(500).json({ error: e?.message || 'write error' }) }
})

router.post('/mkdir', express.json(), async (req, res)=>{
  try{
    const { path: p } = req.body || {}
    const abs = norm(String(p))
    await fsp.mkdir(abs, { recursive: true })
    res.json({ ok:true })
  }catch(e:any){ res.status(500).json({ error: e?.message || 'mkdir error' }) }
})

router.post('/delete', express.json(), async (req, res)=>{
  try{
    const { path: p } = req.body || {}
    const abs = norm(String(p))
    await fsp.rm(abs, { recursive: true, force: true })
    res.json({ ok:true })
  }catch(e:any){ res.status(500).json({ error: e?.message || 'delete error' }) }
})

router.post('/upload', upload.single('file'), async (req, res)=>{
  try{
    const dir = String(req.query.dir || ROOT)
    const abs = norm(dir)
    const f = req.file
    if(!f) return res.status(400).json({ error:'file required' })
    await fsp.mkdir(abs, { recursive: true })
    const dest = path.join(abs, f.originalname)
    await fsp.writeFile(dest, f.buffer)
    res.json({ ok:true, path: dest })
  }catch(e:any){ res.status(500).json({ error: e?.message || 'upload error' }) }
})

router.get('/download', async (req, res)=>{
  try{
    const abs = norm(String(req.query.path))
    res.download(abs)
  }catch(e:any){ res.status(500).json({ error: e?.message || 'download error' }) }
})

export default router
