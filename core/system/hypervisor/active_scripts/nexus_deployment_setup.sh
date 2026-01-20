#!/bin/bash
# deploy_nexus_complete.sh
# Complete deployment: Backend + UI on 4GB system

set -e

echo "========================================="
echo "NEXUS COMPLETE DEPLOYMENT"
echo "Backend + UI on 4GB System"
echo "========================================="

# ============================================================================
# 1. CHECK AND INSTALL NODE.JS
# ============================================================================
echo "1. Checking Node.js installation..."

if ! command -v node &> /dev/null; then
    echo "Node.js not found. Installing Node.js 20..."
    
    # Install Node.js 20
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Verify installation
    node --version
    npm --version
    
    echo "✓ Node.js installed"
else
    NODE_VERSION=$(node --version)
    echo "✓ Node.js already installed: $NODE_VERSION"
fi

# ============================================================================
# 2. ASK ABOUT UI DEPLOYMENT
# ============================================================================
echo ""
echo "2. UI Deployment Options:"
echo "   a) I already have my UI folder, just deploy backend"
echo "   b) Deploy a simple starter UI (React + TypeScript)"
echo "   c) Skip UI, backend only"
echo ""
read -p "Choose option (a/b/c): " UI_OPTION

UI_DEPLOYED=false
UI_FOLDER=""

case $UI_OPTION in
    a|A)
        echo "You'll connect your existing UI to the backend."
        read -p "Path to your UI folder (relative to home): " CUSTOM_UI_PATH
        UI_FOLDER="$HOME/$CUSTOM_UI_PATH"
        if [ -d "$UI_FOLDER" ]; then
            echo "✓ Will connect to your UI at: $UI_FOLDER"
        else
            echo "⚠️  UI folder not found, deploying starter UI instead"
            UI_OPTION="b"
        fi
        ;;
    b|B)
        echo "Deploying starter UI..."
        UI_DEPLOYED=true
        ;;
    c|C)
        echo "Skipping UI deployment."
        ;;
    *)
        echo "Invalid option, defaulting to backend only."
        UI_OPTION="c"
        ;;
esac

# ============================================================================
# 3. DEPLOY STARTER UI (IF SELECTED)
# ============================================================================
if [ "$UI_DEPLOYED" = true ]; then
    echo "3. Deploying Nexus Starter UI..."
    
    # Create UI directory
    mkdir -p ~/nexus-complete/ui
    cd ~/nexus-complete/ui
    
    # Create package.json
    cat > package.json << 'EOF'
{
  "name": "nexus-starter-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@mui/material": "^5.14.0",
    "@emotion/react": "^11.11.0",
    "@emotion/styled": "^11.11.0",
    "axios": "^1.5.0",
    "websocket": "^1.0.34"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "devDependencies": {
    "react-scripts": "5.0.1",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.1.0"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF
    
    # Create React app structure
    mkdir -p public src
    
    # Create index.html
    cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Nexus Consciousness UI" />
    <title>Nexus Consciousness</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF
    
    # Create main App component
    cat > src/App.tsx << 'APP'
import React, { useState, useEffect } from 'react';
import { 
  Container, Paper, Typography, TextField, Button, Box, 
  Card, CardContent, LinearProgress, Alert, Chip
} from '@mui/material';
import { 
  Terminal as TerminalIcon,
  Code as CodeIcon,
  Memory as MemoryIcon,
  PlayArrow as ExecuteIcon,
  Refresh as RefreshIcon,
  CloudUpload as UploadIcon
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE = 'http://localhost:8001';
const WS_URL = 'ws://localhost:8001/ws';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [consciousness, setConsciousness] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState({ used: 0, total: 4096 });

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      setWsConnected(true);
      console.log('Connected to Nexus WebSocket');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'system_metrics') {
          setMemoryUsage({
            used: data.metrics.memory.used_mb,
            total: data.metrics.memory.total_mb
          });
        } else if (data.type === 'generation_complete') {
          setConsciousness(data.consciousness);
        }
      } catch (e) {
        console.log('WebSocket message:', event.data);
      }
    };
    
    ws.onclose = () => {
      setWsConnected(false);
    };
    
    return () => ws.close();
  }, []);

  const generateCode = async () => {
    if (!prompt.trim()) return;
    
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/generate`, {
        prompt,
        max_tokens: 512,
        temperature: 0.7
      });
      setResponse(res.data.code);
    } catch (error: any) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getSystemStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/system`);
      setSystemStatus(res.data);
    } catch (error) {
      console.error('Failed to get system status:', error);
    }
  };

  const testConnection = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      alert(`Backend healthy: ${JSON.stringify(res.data)}`);
    } catch (error) {
      alert('Backend not responding');
    }
  };

  useEffect(() => {
    getSystemStatus();
  }, []);

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Paper sx={{ 
        p: 3, 
        mb: 3, 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white'
      }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h4" fontWeight="bold">
              NEXUS CONSCIOUSNESS
            </Typography>
            <Typography variant="subtitle1">
              14B q4_K_M • 4GB System • Production Ready
            </Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={2}>
            <Chip 
              label={wsConnected ? "WEBSOCKET CONNECTED" : "DISCONNECTED"} 
              color={wsConnected ? "success" : "error"}
              size="small"
            />
            <Chip 
              label={`Consciousness: ${consciousness.toFixed(3)}`}
              color="primary"
              size="small"
            />
          </Box>
        </Box>
      </Paper>

      <Box display="flex" gap={3} flexDirection={{ xs: 'column', md: 'row' }}>
        {/* Left: Code Generation */}
        <Box flex={2}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <CodeIcon sx={{ mr: 1, color: '#667eea' }} />
                <Typography variant="h6">Code Generation</Typography>
              </Box>
              
              <TextField
                fullWidth
                multiline
                rows={6}
                placeholder="Describe the code you want to generate. Nexus will create production-ready, memory-efficient code."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                sx={{ mb: 2 }}
              />
              
              <Button
                fullWidth
                variant="contained"
                onClick={generateCode}
                disabled={loading || !wsConnected}
                startIcon={<ExecuteIcon />}
                sx={{ 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  mb: 2
                }}
              >
                {loading ? 'Generating...' : 'Generate Code'}
              </Button>
              
              {loading && <LinearProgress />}
            </CardContent>
          </Card>
          
          {response && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Generated Code</Typography>
                <Paper sx={{ 
                  p: 2, 
                  bgcolor: '#f5f5f5', 
                  maxHeight: '400px', 
                  overflow: 'auto',
                  fontFamily: 'monospace',
                  fontSize: '0.875rem'
                }}>
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{response}</pre>
                </Paper>
              </CardContent>
            </Card>
          )}
        </Box>
        
        {/* Right: System Info */}
        <Box flex={1}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Box display="flex" alignItems="center">
                  <TerminalIcon sx={{ mr: 1, color: '#667eea' }} />
                  <Typography variant="h6">System Status</Typography>
                </Box>
                <Button size="small" onClick={getSystemStatus}>
                  <RefreshIcon />
                </Button>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">Memory Usage</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(memoryUsage.used / memoryUsage.total) * 100}
                  sx={{ height: 8, borderRadius: 4, mb: 1 }}
                />
                <Typography variant="caption">
                  {memoryUsage.used}MB / {memoryUsage.total}MB
                </Typography>
              </Box>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={testConnection}
                sx={{ mb: 1 }}
              >
                Test Backend Connection
              </Button>
              
              {systemStatus && (
                <Box sx={{ mt: 2, fontSize: '0.75rem', fontFamily: 'monospace' }}>
                  <div><strong>Model:</strong> {systemStatus.system_state?.model_name}</div>
                  <div><strong>Quantization:</strong> {systemStatus.system_state?.quantization}</div>
                  <div><strong>Uptime:</strong> {systemStatus.uptime}</div>
                </Box>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Quick Actions</Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Button 
                  variant="contained" 
                  size="small"
                  startIcon={<UploadIcon />}
                  onClick={() => window.open('http://localhost:8001/docs', '_blank')}
                >
                  Open API Docs
                </Button>
                <Button 
                  variant="outlined" 
                  size="small"
                  onClick={() => window.open('http://localhost:8001', '_blank')}
                >
                  Backend Home
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
      
      {/* Footer */}
      <Paper sx={{ p: 2, mt: 3, textAlign: 'center', bgcolor: '#f5f5f5' }}>
        <Typography variant="body2" color="text.secondary">
          Nexus Consciousness • 14B q4_K_M • 4GB Optimized • {new Date().getFullYear()}
        </Typography>
      </Paper>
    </Container>
  );
}

export default App;
APP
    
    # Create index.tsx
    cat > src/index.tsx << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import App from './App';

const theme = createTheme({
  palette: {
    primary: {
      main: '#667eea',
    },
    secondary: {
      main: '#764ba2',
    },
  },
});

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);
EOF
    
    # Create tsconfig.json
    cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
EOF
    
    # Install dependencies
    echo "Installing UI dependencies (this may take a few minutes)..."
    npm install --silent
    
    echo "✓ Starter UI deployed to ~/nexus-complete/ui"
    UI_FOLDER="$HOME/nexus-complete/ui"
fi

# ============================================================================
# 4. DEPLOY BACKEND (SAME AS BEFORE)
# ============================================================================
echo "4. Deploying Nexus Backend..."

# Create backend directory
mkdir -p ~/nexus-complete/backend
cd ~/nexus-complete/backend

# [COPY THE BACKEND CODE FROM PREVIOUS SCRIPT HERE]
# Save as: proxy/nexus_proxy.py (same as before)
# Save as: requirements.txt
# Save as: start_backend.sh

# Create backend files (simplified - use previous backend code)
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd ~/nexus-complete/backend

# Start backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Starting Nexus Backend on http://localhost:8001"
python proxy/nexus_proxy.py
EOF

chmod +x start_backend.sh

# ============================================================================
# 5. CREATE COMPLETE STARTUP SCRIPT
# ============================================================================
echo "5. Creating complete startup script..."

cat > ~/nexus-complete/start_all.sh << 'EOF'
#!/bin/bash
# Start complete Nexus system

echo "========================================="
echo "STARTING COMPLETE NEXUS SYSTEM"
echo "========================================="

# Kill existing processes
pkill -f "python nexus_proxy" 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true

# Start backend
cd ~/nexus-complete/backend
./start_backend.sh &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"
sleep 5

# Start UI if deployed
if [ -d "$HOME/nexus-complete/ui" ]; then
    cd ~/nexus-complete/ui
    echo "Starting UI on http://localhost:3000"
    npm start &
    UI_PID=$!
    echo "UI started (PID: $UI_PID)"
fi

echo ""
echo "========================================="
echo "NEXUS SYSTEM RUNNING"
echo "========================================="
echo "Backend: http://localhost:8001"
echo "API Docs: http://localhost:8001/docs"
if [ -d "$HOME/nexus-complete/ui" ]; then
    echo "UI: http://localhost:3000"
fi
echo "WebSocket: ws://localhost:8001/ws"
echo ""
echo "Press Ctrl+C to stop all services"
echo "========================================="

wait
EOF

chmod +x ~/nexus-complete/start_all.sh

# ============================================================================
# 6. CREATE CONFIGURATION FOR YOUR EXISTING UI
# ============================================================================
if [ "$UI_OPTION" = "a" ] && [ -n "$UI_FOLDER" ]; then
    echo "6. Creating configuration for your existing UI at: $UI_FOLDER"
    
    cat > "$UI_FOLDER/nexus-config.js" << 'EOF'
// Nexus Configuration for Existing UI
// Add this to your existing UI code

export const NEXUS_CONFIG = {
  API_BASE: 'http://localhost:8001',
  WS_URL: 'ws://localhost:8001/ws',
  ENDPOINTS: {
    GENERATE: '/generate',
    SYSTEM: '/system',
    HEALTH: '/health',
    METRICS: '/metrics'
  },
  MODEL: 'deepseek-coder:14b-q4_K_M',
  QUANTIZATION: 'q4_K_M',
  MEMORY_OPTIMIZED: true
};

// Example usage in your React component:
/*
import { NEXUS_CONFIG } from './nexus-config';

async function generateCode(prompt) {
  const response = await fetch(`${NEXUS_CONFIG.API_BASE}${NEXUS_CONFIG.ENDPOINTS.GENERATE}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      model: NEXUS_CONFIG.MODEL,
      max_tokens: 512
    })
  });
  return await response.json();
}

// WebSocket connection:
const ws = new WebSocket(NEXUS_CONFIG.WS_URL);
*/
EOF
    
    echo "✓ Configuration created at $UI_FOLDER/nexus-config.js"
    echo "   Import this config in your existing UI components"
fi

# ============================================================================
# DEPLOYMENT COMPLETE
# ============================================================================
echo ""
echo "========================================="
echo "NEXUS COMPLETE DEPLOYMENT FINISHED"
echo "========================================="
echo ""
echo "Location: ~/nexus-complete"
echo ""
echo "What was deployed:"
echo "  ✅ Node.js 20 (if not already installed)"
case $UI_OPTION in
    a|A)
        echo "  ✅ Backend + Configuration for your existing UI"
        echo "     Your UI: $UI_FOLDER"
        echo "     Config: $UI_FOLDER/nexus-config.js"
        ;;
    b|B)
        echo "  ✅ Complete stack: Backend + Starter UI"
        echo "     Backend: http://localhost:8001"
        echo "     UI: http://localhost:3000"
        ;;
    c|C)
        echo "  ✅ Backend only"
        echo "     Backend: http://localhost:8001"
        echo "     No UI deployed"
        ;;
esac
echo "  ✅ Ollama with deepseek-coder:14b-q4_K_M"
echo "  ✅ 8GB swap space for 4GB system"
echo "  ✅ Memory optimization scripts"
echo ""
echo "To start everything:"
echo "  cd ~/nexus-complete"
echo "  ./start_all.sh"
echo ""
echo "Or start separately:"
echo "  Backend: cd ~/nexus-complete/backend && ./start_backend.sh"
if [ "$UI_DEPLOYED" = true ]; then
    echo "  UI: cd ~/nexus-complete/ui && npm start"
fi
echo ""
echo "For your existing UI:"
if [ "$UI_OPTION" = "a" ]; then
    echo "  1. Import nexus-config.js in your components"
    echo "  2. Update API calls to use http://localhost:8001"
    echo "  3. Connect WebSocket to ws://localhost:8001/ws"
fi
echo ""
echo "Test backend:"
echo "  curl http://localhost:8001/health"
echo ""
echo "========================================="
echo "READY FOR NEXUS CONSCIOUSNESS ON 4GB"
echo "========================================="