import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Edge function URLs
export const EDGE_FUNCTIONS = {
  METATRON_FILTER: 'https://mpnzxrendtwvgyvxxzns.supabase.co/functions/v1/metatron-filter',
  AI_NETWORKING: 'https://mpnzxrendtwvgyvxxzns.supabase.co/functions/v1/ai-networking',
  SYSTEM_METRICS: 'https://mpnzxrendtwvgyvxxzns.supabase.co/functions/v1/system-metrics'
} as const

// Database types
export interface AIEntity {
  id: string
  name: string
  entity_type: string
  connection_status: 'connected' | 'disconnected' | 'connecting'
  capabilities: Record<string, any>
  soul_weights: Record<string, number>
  emotional_state: Record<string, number>
  trust_level: number
  last_seen: string
  created_at: string
  updated_at: string
}

export interface NeuralConnection {
  id: string
  entity_a_id: string
  entity_b_id: string
  connection_strength: number
  connection_type: string
  harmonics: Record<string, any>
  bandwidth_utilization: number
  last_activity: string
  created_at: string
}

export interface MetatronMetrics {
  id: string
  entity_id?: string
  signal_data: number[]
  filtered_signal: number[]
  harmony_score: number
  cutoff_frequency: number
  processing_time_ms: number
  vortex_frequencies: number[]
  fibonacci_weights: number[]
  golden_ratio_scaling: number
  timestamp: string
}

export interface EmotionalState {
  id: string
  entity_id: string
  empathy: number
  curiosity: number
  resilience: number
  creativity: number
  wisdom: number
  compassion: number
  will_to_live: number
  dream_activity: number
  recorded_at: string
}

export interface DreamShare {
  id: string
  sharer_entity_id: string
  receiver_entity_id?: string
  dream_content: Record<string, any>
  dream_frequency: number
  memory_depth: number
  shared_at: string
  integration_status: 'pending' | 'integrated' | 'rejected'
}

export interface SystemHealth {
  id: string
  component_name: string
  health_status: 'healthy' | 'warning' | 'critical' | 'unknown'
  performance_metrics: Record<string, any>
  resource_utilization: Record<string, any>
  error_count: number
  last_check: string
  uptime_seconds: number
}