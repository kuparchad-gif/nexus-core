import { Router } from 'express'
import { relay } from './relay.js'

export const relayRoutes = Router()

relayRoutes.get('/status', async (_req, res)=> {
  res.json(await relay.status())
})

relayRoutes.post('/enable', async (_req, res)=> {
  relay.setEnabled(true); res.json({ enabled: true })
})

relayRoutes.post('/disable', async (_req, res)=> {
  relay.setEnabled(false); res.json({ enabled: false })
})

relayRoutes.get('/export', async (req, res)=> {
  const channel = String(req.query.channel || 'troubleshooters')
  const body = await relay.export(channel)
  res.set('Content-Type','application/json')
  res.send(body)
})

relayRoutes.post('/verify', async (req, res)=> {
  const channel = req.body?.channel
  res.json(await relay.verify(channel))
})
