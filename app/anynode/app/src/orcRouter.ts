import { Router } from 'express'
import { createProxyMiddleware } from 'http-proxy-middleware'

/**
 * Reverse proxy to upstream services (secure by default).
 * Env:
 *  - ORC_TARGET: base URL for your router/orchestrator (e.g., http://10.0.0.5:7000)
 *  - LMSTUDIO_TARGET: base URL for LM Studio API (e.g., http://10.0.0.8:1234)
 */
export const orcRouter = Router()

function mkProxy(targetEnv: string, basePath: string){
  const target = process.env[targetEnv]
  if (!target) return null
  return createProxyMiddleware(basePath, {
    target,
    changeOrigin: true,
    ws: true,
    secure: false,
    pathRewrite: (path) => path.replace(new RegExp('^' + basePath), '/'),
    onProxyReq: (proxyReq, req, res)=>{
      proxyReq.setHeader('x-forwarded-host', req.headers.host || '')
      proxyReq.setHeader('x-orc-proxy', '1')
    }
  })
}

const orc = mkProxy('ORC_TARGET', '/orc')
const lm = mkProxy('LMSTUDIO_TARGET', '/lmstudio')

if (orc) orcRouter.use(orc)
if (lm) orcRouter.use(lm)
