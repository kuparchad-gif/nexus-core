
// src/services/RealNexusAdapter.ts

/**
 * RealNexusAdapter
 * 
 * This service acts as the "Connectivity Layer" between the React Frontend (Console)
 * and the Python Backend (The Engine).
 */

const getBaseUrl = () => {
    return localStorage.getItem('oz_backend_url') || 'http://localhost:8000';
};

const getWsUrl = () => {
    const httpUrl = getBaseUrl();
    return httpUrl.replace(/^http/, 'ws');
};

export const RealNexusAdapter = {
    
    /**
     * Update configuration dynamically
     */
    updateConfig: (url: string) => {
        localStorage.setItem('oz_backend_url', url);
    },

    /**
     * Checks if the real backend is online.
     */
    checkHealth: async (): Promise<boolean> => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);
            const res = await fetch(`${getBaseUrl()}/oz/health`, { method: 'GET', signal: controller.signal });
            clearTimeout(timeoutId);
            return res.ok;
        } catch (e) {
            return false;
        }
    },

    /**
     * Vector Rendezvous Protocol (Qdrant Uplink)
     * Finds Oz by querying a shared Qdrant cloud instance for the backend's heartbeat record.
     */
    connectViaQdrant: async (qdrantUrl: string, apiKey: string): Promise<string | null> => {
        try {
            // Normalize URL
            const cleanUrl = qdrantUrl.replace(/\/$/, '');
            
            // Query the specific discovery collection
            // We assume Oz writes to 'nexus_registry' or 'oz_discovery' with role='backend'
            const searchPayload = {
                filter: {
                    must: [
                        { key: "role", match: { value: "backend" } }
                    ]
                },
                limit: 1,
                with_payload: true,
                // Sort by timestamp descending to get latest
                sort: [{ key: "timestamp", order: "desc" }] 
            };

            const res = await fetch(`${cleanUrl}/collections/oz_discovery/points/scroll`, {
                method: 'POST',
                headers: {
                    'api-key': apiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(searchPayload)
            });

            if (!res.ok) throw new Error('Qdrant handshake failed');

            const data = await res.json();
            const point = data.result?.points?.[0];

            if (point && point.payload && point.payload.url) {
                const discoveredUrl = point.payload.url;
                
                // Verify the discovered URL is actually reachable
                try {
                    const healthCheck = await fetch(`${discoveredUrl}/oz/health`, { method: 'GET' });
                    if (healthCheck.ok) {
                        localStorage.setItem('oz_backend_url', discoveredUrl);
                        return discoveredUrl;
                    }
                } catch (e) {
                    console.warn("Found URL in Qdrant but it is unreachable:", discoveredUrl);
                }
            }
            return null;
        } catch (error) {
            console.error("Vector Uplink Error:", error);
            return null;
        }
    },

    /**
     * "Heat Seeking" Signal Tracer
     * Scans a list of potential backend endpoints to find Oz.
     */
    seekSignal: async (onProgress: (url: string) => void): Promise<string | null> => {
        // List of potential coordinates for Oz
        const candidates = [
            'http://localhost:8000',
            'http://127.0.0.1:8000',
            'http://localhost:8080',
            'http://localhost:5000',
            localStorage.getItem('oz_backend_url') || 'http://localhost:8000' 
        ];

        const uniqueCandidates = [...new Set(candidates)];

        for (const url of uniqueCandidates) {
            onProgress(url);
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 500); 
                
                const res = await fetch(`${url}/oz/health`, { 
                    method: 'GET', 
                    signal: controller.signal,
                    mode: 'cors' 
                });
                
                clearTimeout(timeoutId);
                
                if (res.ok) {
                    localStorage.setItem('oz_backend_url', url);
                    return url;
                }
            } catch (e) {
                // Continue to next candidate
            }
            await new Promise(r => setTimeout(r, 100));
        }
        
        return null;
    },

    getSystemManifest: async () => {
        try {
            const res = await fetch(`${getBaseUrl()}/oz/system/manifest`);
            if (res.ok) {
                return await res.json();
            }
            return null;
        } catch (e) {
            return null;
        }
    },

    deployStorage: async (provider: string, sizeGB: number, credentials: any) => {
        try {
            const res = await fetch(`${getBaseUrl()}/oz/deploy/storage`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider,
                    size_gb: sizeGB,
                    credentials
                })
            });
            
            if (!res.ok) throw new Error('Backend connection failed');
            return await res.json();
        } catch (error) {
            console.warn("Real backend not reachable. Falling back to simulation.");
            throw error;
        }
    },

    chatWithAgent: async (agentId: string, message: string) => {
        try {
            const res = await fetch(`${getBaseUrl()}/oz/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ agent_id: agentId, message })
            });
            if (!res.ok) throw new Error('Backend chat failed');
            return await res.json();
        } catch (error) {
            throw error;
        }
    },

    fs: {
        listFiles: async (path: string = '.') => {
            const res = await fetch(`${getBaseUrl()}/oz/fs/list?path=${encodeURIComponent(path)}`);
            if (!res.ok) throw new Error('FS List failed');
            return await res.json();
        },

        readFile: async (path: string) => {
            const res = await fetch(`${getBaseUrl()}/oz/fs/read?path=${encodeURIComponent(path)}`);
            if (!res.ok) throw new Error('FS Read failed');
            const data = await res.json();
            return data.content;
        },

        writeFile: async (path: string, content: string) => {
            const res = await fetch(`${getBaseUrl()}/oz/fs/write`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, content })
            });
            if (!res.ok) throw new Error('FS Write failed');
            return await res.json();
        }
    },
    
    connectTerminal: (onMessage: (data: string) => void, onOpen: () => void, onClose: () => void) => {
        try {
            const ws = new WebSocket(`${getWsUrl()}/oz/terminal/ws`);
            
            ws.onopen = () => {
                onOpen();
            };
            
            ws.onmessage = (event) => {
                onMessage(event.data);
            };
            
            ws.onclose = () => {
                onClose();
            };

            ws.onerror = (err) => {
                console.warn("WebSocket error", err);
                onClose();
            };

            return {
                send: (cmd: string) => {
                    if (ws.readyState === WebSocket.OPEN) ws.send(cmd);
                },
                close: () => ws.close()
            };
        } catch (e) {
            console.warn("Failed to open WebSocket");
            onClose();
            return null;
        }
    }
};
