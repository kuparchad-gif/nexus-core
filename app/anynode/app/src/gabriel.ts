import dgram from 'dgram'
import net from 'net'
import { WebSocketServer } from 'ws'
import type { Server as HTTPServer } from 'http'
import { relay } from './relay.js'

/**
 * Gabriel: lightweight network receivers for JSON frames.
 * Env:
 *  - GABRIEL_UDP_PORT (default: 9400/udp)
 *  - GABRIEL_TCP_PORT (default: 9500/tcp)
 */
export function startGabriel(server: HTTPServer){
  const udpPort = parseInt(process.env.GABRIEL_UDP_PORT || '9400', 10)
  const tcpPort = parseInt(process.env.GABRIEL_TCP_PORT || '9500', 10)

  const wss = new WebSocketServer({ server, path: '/ws/gabriel' })

  // UDP
  const udp = dgram.createSocket('udp4')
  udp.on('message', async (msg, rinfo)=>{
    try{
      const txt = msg.toString('utf-8')
      const obj = JSON.parse(txt)
      const frame = { type: 'udp', from: rinfo.address+':'+rinfo.port, data: obj, ts: Date.now() }
      relay.append('gabriel', frame).catch(()=>{})
      const payload = JSON.stringify({ channel:'gabriel', frame })
      ;(wss.clients||[] as any).forEach((ws:any)=> { try{ ws.send(payload) }catch{} })
    }catch{}
  })
  udp.bind(udpPort)

  // TCP
  const tcp = net.createServer((sock)=>{
    let buf = ''
    sock.on('data', (chunk)=>{
      buf += chunk.toString('utf-8')
      let idx
      while((idx = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0, idx); buf = buf.slice(idx+1)
        if (!line.trim()) continue
        try{
          const obj = JSON.parse(line)
          const frame = { type: 'tcp', from: sock.remoteAddress+':'+sock.remotePort, data: obj, ts: Date.now() }
          relay.append('gabriel', frame).catch(()=>{})
          const payload = JSON.stringify({ channel:'gabriel', frame })
          ;(wss.clients||[] as any).forEach((ws:any)=> { try{ ws.send(payload) }catch{} })
        }catch{}
      }
    })
  })
  tcp.listen(tcpPort)

  return { udpPort, tcpPort }
}
