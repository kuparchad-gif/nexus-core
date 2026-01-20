import { LogEntry, Service } from "../types";

export const mockLogs: LogEntry[] = Array.from({ length: 100 }, (_, i) => {
  const services = ['Firewall', 'Loki', 'Viren', 'Lillith', 'Pulse'];
  const service = services[i % services.length];
  const levels: LogEntry['level'][] = ['INFO', 'INFO', 'INFO', 'WARN', 'ERROR'];
  const level = levels[Math.floor(Math.random() * levels.length)];
  
  return {
    timestamp: new Date(Date.now() - i * 60000).toISOString(),
    service: service,
    level,
    message: `Service operation ${i} completed with status ${level}.`
  }
});

export const mockFiles = [
  { name: 'loki-2024-07-28.log.gz', size: '15.2 MB', modified: '2024-07-28 23:59:00' },
  { name: 'loki-2024-07-27.log.gz', size: '14.8 MB', modified: '2024-07-27 23:59:00' },
  { name: 'viren-ops-trace.json', size: '2.1 MB', modified: '2024-07-28 14:30:10' },
  { name: 'system_boot.log', size: '512 KB', modified: '2024-07-20 01:00:00' },
];

export const getMockViraaAnalysis = (service: Service | null): Promise<string> => {
  if (!service) return Promise.resolve("No service selected for analysis.");
  
  const analysis = `Viraa Analysis for ${service.name}:
- Status: ${service.status.toUpperCase()}. Current CPU usage is at ${service.metrics.cpu}% and Memory at ${service.metrics.memory}%.
- LLM Config: Using model ${service.llmConfig.model} at endpoint ${service.llmConfig.endpoint}.
- Qdrant DB: Connected to collection '${service.qdrantConfig.collection}'.
- Recommendation: System performing within expected parameters. No immediate action required.`;
  
  return new Promise(resolve => setTimeout(() => resolve(analysis), 1000));
};