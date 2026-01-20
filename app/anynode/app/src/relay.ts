import fs from 'fs'
import path from 'path'
import crypto from 'crypto'
import { Storage } from '@google-cloud/storage'

export type RelayRecord = {
  index: number
  ts: number
  channel: string
  payloadHash: string
  prevHash: string
  hash: string
}

interface RelayBackend {
  init(): Promise<void>
  append(channel: string, record: RelayRecord): Promise<void>
  head(channel: string): Promise<RelayRecord | null>
  export(channel: string): Promise<string>
  verify(channel: string): Promise<{ ok: boolean, count: number, at?: number, expect?: string, got?: string, head?: string }>
  listChannels(): Promise<string[]>
}

const makeRecord = (channel: string, prev: RelayRecord | null, payload: any): RelayRecord => {
  const payloadStr = JSON.stringify(payload)
  const payloadHash = crypto.createHash('sha256').update(payloadStr).digest('hex')
  const prevHash = prev?.hash || 'GENESIS'
  const index = (prev?.index ?? -1) + 1
  const ts = Date.now()
  const header = `${index}|${ts}|${channel}|${payloadHash}|${prevHash}`
  const hash = crypto.createHash('sha256').update(header).digest('hex')
  return { index, ts, channel, payloadHash, prevHash, hash }
}

/** ---------- FS BACKEND ---------- */
class FSBackend implements RelayBackend {
  private dir: string
  private cacheHeads: Record<string, RelayRecord | null> = {}
  constructor(dir: string){ this.dir = dir }
  async init(){
    if (!fs.existsSync(this.dir)) fs.mkdirSync(this.dir, { recursive: true })
    // no-op preload; heads lazy-loaded
  }
  private fileFor(channel: string){ return path.join(this.dir, channel + '.chain.jsonl') }
  async append(channel: string, record: RelayRecord){
    fs.appendFile(this.fileFor(channel), JSON.stringify(record) + '\n', ()=>{})
    this.cacheHeads[channel] = record
  }
  async head(channel: string){
    if (this.cacheHeads[channel] !== undefined) return this.cacheHeads[channel]
    const f = this.fileFor(channel)
    if (!fs.existsSync(f)){ this.cacheHeads[channel] = null; return null }
    const data = fs.readFileSync(f, 'utf-8').trim()
    if (!data){ this.cacheHeads[channel] = null; return null }
    const lines = data.split(/\r?\n/).filter(Boolean)
    const rec: RelayRecord = JSON.parse(lines[lines.length-1])
    this.cacheHeads[channel] = rec
    return rec
  }
  async export(channel: string){
    const f = this.fileFor(channel)
    if (!fs.existsSync(f)) return ''
    return fs.readFileSync(f, 'utf-8')
  }
  async verify(channel: string){
    const f = this.fileFor(channel)
    if (!fs.existsSync(f)) return { ok: true, count: 0 }
    const lines = fs.readFileSync(f, 'utf-8').trim().split(/\r?\n/).filter(Boolean)
    let prevHash = 'GENESIS'
    for (let i=0;i<lines.length;i++){
      const rec = JSON.parse(lines[i])
      const header = `${rec.index}|${rec.ts}|${rec.channel}|${rec.payloadHash}|${prevHash}`
      const expect = crypto.createHash('sha256').update(header).digest('hex')
      if (expect !== rec.hash) return { ok: false, at: i, expect, got: rec.hash, count: i }
      prevHash = rec.hash
    }
    return { ok: true, count: lines.length, head: prevHash }
  }
  async listChannels(){
    if (!fs.existsSync(this.dir)) return []
    return fs.readdirSync(this.dir).filter(n=>n.endsWith('.chain.jsonl')).map(n=>n.replace('.chain.jsonl',''))
  }
}

/** ---------- GCS BACKEND ---------- */
class GCSBackend implements RelayBackend {
  private bucketName: string
  private prefix: string
  private storage: Storage
  constructor(bucket: string, prefix: string = ''){
    this.bucketName = bucket
    this.prefix = prefix.replace(/\/+$/,'')
    this.storage = new Storage()
  }
  async init(){ /* relies on ADC or GOOGLE_APPLICATION_CREDENTIALS */ }
  private key(channel: string){
    const base = `${channel}.chain.jsonl`
    return this.prefix ? `${this.prefix}/${base}` : base
  }
  async append(channel: string, record: RelayRecord){
    const file = this.storage.bucket(this.bucketName).file(this.key(channel))
    const line = JSON.stringify(record) + '\n'
    // Compose append: try read -> append -> write. For simplicity here, read-modify-write.
    let prev = ''
    try{
      const [exists] = await file.exists()
      if (exists){
        const [buf] = await file.download()
        prev = buf.toString('utf-8')
      }
    }catch{}
    await file.save(prev + line, { contentType: 'application/json', resumable: false })
  }
  async head(channel: string){
    const exp = await this.export(channel)
    if (!exp) return null
    const lines = exp.trim().split(/\r?\n/).filter(Boolean)
    if (!lines.length) return null
    return JSON.parse(lines[lines.length-1])
  }
  async export(channel: string){
    const file = this.storage.bucket(this.bucketName).file(this.key(channel))
    const [exists] = await file.exists()
    if (!exists) return ''
    const [buf] = await file.download()
    return buf.toString('utf-8')
  }
  async verify(channel: string){
    const content = await this.export(channel)
    if (!content) return { ok: true, count: 0 }
    const lines = content.trim().split(/\r?\n/).filter(Boolean)
    let prevHash = 'GENESIS'
    for (let i=0;i<lines.length;i++){
      const rec = JSON.parse(lines[i])
      const header = `${rec.index}|${rec.ts}|${rec.channel}|${rec.payloadHash}|${prevHash}`
      const expect = crypto.createHash('sha256').update(header).digest('hex')
      if (expect !== rec.hash) return { ok: false, at: i, expect, got: rec.hash, count: i }
      prevHash = rec.hash
    }
    return { ok: true, count: lines.length, head: prevHash }
  }
  async listChannels(){
    const [files] = await this.storage.bucket(this.bucketName).getFiles({ prefix: this.prefix ? this.prefix + '/' : '' })
    return files
      .map(f => f.name)
      .filter(n => n.endsWith('.chain.jsonl'))
      .map(n => n.split('/').pop()!.replace('.chain.jsonl',''))
  }
}

/** ---------- Manager ---------- */
class RelayManager {
  private backend: RelayBackend
  private channels = ['troubleshooters','training','kubes','gabriel']
  private enabled = true

  constructor(){
    const mode = (process.env.RELAY_ADAPTER || 'fs').toLowerCase()
    if (mode === 'gcs'){
      const bucket = process.env.RELAY_GCS_BUCKET
      if (!bucket) throw new Error('RELAY_GCS_BUCKET required for gcs adapter')
      const prefix = process.env.RELAY_GCS_PREFIX || 'relays'
      this.backend = new GCSBackend(bucket, prefix)
    } else {
      const dir = path.join(process.cwd(), 'data', 'relays')
      this.backend = new FSBackend(dir)
    }
  }

  async init(){ await this.backend.init() }

  setEnabled(v: boolean){ this.enabled = v }
  isEnabled(){ return this.enabled }

  async append(channel: string, payload: any){
    if (!this.enabled) return
    const prev = await this.backend.head(channel)
    const rec = makeRecord(channel, prev, payload)
    await this.backend.append(channel, rec)
  }

  async status(){
    const out: Record<string, any> = {}
    for (const ch of this.channels){
      const h = await this.backend.head(ch)
      out[ch] = h ? { index: h.index, hash: h.hash, ts: h.ts } : { index: -1, hash: null, ts: null }
    }
    return { enabled: this.enabled, heads: out }
  }

  async export(channel: string){
    return await this.backend.export(channel)
  }

  async verify(channel?: string){
    const verifyOne = async (ch: string) => await this.backend.verify(ch)
    if (channel){
      const res = await verifyOne(channel)
      return { [channel]: res }
    } else {
      const out: Record<string, any> = {}
      for (const ch of this.channels){
        out[ch] = await verifyOne(ch)
      }
      return out
    }
  }
}

export const relay = new RelayManager()
