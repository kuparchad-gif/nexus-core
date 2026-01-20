import { Router } from 'express'

export type NodeInfo = {
  id: string
  name: string
  role: 'worker'|'control'
  cpu: number
  mem: number
  pods: number
  status: 'green'|'yellow'|'red'
}

export type Cluster = {
  name: string
  nodes: NodeInfo[]
}

export const KubesState: { cogni: Cluster, acidemi: Cluster } = {
  cogni: {
    name: 'CogniKubes',
    nodes: [
      { id: 'c1', name: 'cogni-1', role: 'control', cpu: 32, mem: 48, pods: 56, status: 'green' },
      { id: 'c2', name: 'cogni-2', role: 'worker',  cpu: 18, mem: 36, pods: 44, status: 'green' },
      { id: 'c3', name: 'cogni-3', role: 'worker',  cpu: 22, mem: 52, pods: 60, status: 'yellow' }
    ]
  },
  acidemi: {
    name: 'AcidemiKubes',
    nodes: [
      { id: 'a1', name: 'acidemi-1', role: 'control', cpu: 28, mem: 40, pods: 49, status: 'green' },
      { id: 'a2', name: 'acidemi-2', role: 'worker',  cpu: 35, mem: 58, pods: 72, status: 'yellow' },
      { id: 'a3', name: 'acidemi-3', role: 'worker',  cpu: 55, mem: 66, pods: 85, status: 'red' }
    ]
  }
}

export const kubes = Router()

kubes.get('/state', (_req, res)=>{
  res.json(KubesState)
})
