import React, { useEffect, useState } from 'react'

type Prompt = { id: string; title: string; system: string; user_template: string; meta?: any }
type RouteReq = { prompt: string; mode: string; k: number; combiner: string }
type ExpertCall = { name: string; url: string; score: number }
type RouteResp = { prompt: string; mode: string; selected: ExpertCall[]; scores: Record<string,number>; outputs: Record<string,any>; combined: any }

const API = (path:string) => (window.location.origin.includes(':5173') ? 'http://localhost:8000' : '/api') + path

export default function App(){
  const [prompt,setPrompt] = useState('simulate the network and tell me the mean response time')
  const [mode,setMode] = useState<'best'|'top_k'|'broadcast'>('top_k')
  const [k,setK] = useState(2)
  const [combiner,setCombiner] = useState<'first'|'concat'|'vote'>('concat')
  const [resp,setResp] = useState<RouteResp|null>(null)
  const [busy,setBusy] = useState(false)
  const [prompts,setPrompts] = useState<Prompt[]>([])
  const [selectedPrompt,setSelectedPrompt] = useState<string>('')

  async function loadPrompts(){
    try{
      const r = await fetch(API('/v1/prompts'))
      const j = await r.json()
      setPrompts(j.prompts || [])
    }catch{}
  }
  useEffect(()=>{ loadPrompts() },[])

  function applyPrompt(id: string){
    const p = prompts.find(x=>x.id===id)
    if(!p) return
    setSelectedPrompt(id)
    const templ = p.user_template || '{input}'
    const rendered = templ.replace('{input}', prompt)
    setPrompt(rendered)
  }

  async function run(){
    setBusy(true)
    try{
      const r = await fetch(API('/v1/route'),{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({prompt, mode, k, combiner} as RouteReq)
      })
      const j: RouteResp = await r.json()
      setResp(j)
    }finally{
      setBusy(false)
    }
  }

  async function savePrompt(){
    const id = prompt.slice(0,24).toLowerCase().replace(/[^a-z0-9]+/g,'-') || 'custom'
    const spec: Prompt = { id, title: 'Custom '+ new Date().toISOString(), system: '', user_template: prompt }
    await fetch(API('/v1/prompts/'+id),{method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(spec)})
    await loadPrompts()
    setSelectedPrompt(id)
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-3xl font-semibold mb-2">Nexus MoE</h1>
      <p className="text-sm text-gray-600 mb-6">Mixture-of-experts gateway with LM Toolbox & Prompt Registry</p>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 card p-4">
          <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} rows={8}
            className="w-full p-3 border rounded-2xl border-silver focus:outline-purple" />
          <div className="flex items-center gap-3 mt-3">
            <select value={mode} onChange={e=>setMode(e.target.value as any)} className="btn">
              <option>best</option><option>top_k</option><option>broadcast</option>
            </select>
            {mode==='top_k' && <input type="number" min={1} max={5} value={k} onChange={e=>setK(parseInt(e.target.value)||1)} className="btn w-24" />}
            <select value={combiner} onChange={e=>setCombiner(e.target.value as any)} className="btn">
              <option>first</option><option>concat</option><option>vote</option>
            </select>
            <button className="btn btn-primary" onClick={run} disabled={busy}>{busy ? 'Routing…' : 'Route'}</button>
            <button className="btn" onClick={savePrompt}>Save as Prompt</button>
          </div>
        </div>

        <div className="card p-4">
          <h2 className="font-semibold mb-2">Prompts</h2>
          <select className="w-full btn" value={selectedPrompt} onChange={e=>applyPrompt(e.target.value)}>
            <option value="">-- choose --</option>
            {prompts.map(p=>(<option key={p.id} value={p.id}>{p.title}</option>))}
          </select>
          <p className="text-xs text-gray-500 mt-2">Prompts are persisted on the gateway volume.</p>
        </div>
      </div>

      {resp && (
        <div className="grid md:grid-cols-3 gap-4 mt-6">
          <div className="card p-4 md:col-span-1">
            <h2 className="font-semibold mb-2">Gating Scores</h2>
            <ul className="space-y-2">
              {Object.entries(resp.scores).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <li key={k} className="flex items-center gap-2">
                  <span className="badge w-20 text-center">{k}</span>
                  <div className="flex-1 bg-labwhite rounded-2xl h-3">
                    <div className="h-3 rounded-2xl" style={{width: `${Math.round(v*100)}%`, background:'#7E57C2'}} />
                  </div>
                  <span className="text-xs w-10 text-right">{(v*100|0)}%</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="card p-4 md:col-span-2">
            <h2 className="font-semibold mb-2">Responses</h2>
            <div className="grid gap-3">
              {resp.selected.map(s => (
                <div key={s.name} className="border rounded-2xl p-3">
                  <div className="text-sm text-gray-600 mb-1">{s.name} • {(s.score*100|0)}%</div>
                  <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(resp.outputs[s.name], null, 2)}</pre>
                </div>
              ))}
            </div>
            <h3 className="font-semibold mt-4 mb-1">Combined</h3>
            <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(resp.combined, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  )
}
