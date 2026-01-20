#!/usr/bin/env python3
"""
NEXUS WEB CONSOLE - Modern Web Interface for Nexus CLI
Real-time web interface with SSE, WebSocket, and beautiful UI components
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

# Import your existing CLI functionality
from hybrid_heroku_cli import HybridHerokuCLI, DynoManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("nexus-web-console")

class WebConsoleManager:
    """Manages web console sessions and real-time communication"""
    
    def __init__(self):
        self.cli = HybridHerokuCLI()
        self.active_connections: List[WebSocket] = []
        self.command_history: List[Dict] = []
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
        
        # Send welcome message
        await self._send_personal_message({
            "type": "system",
            "message": "🔮 Welcome to Nexus Web Console!",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        # Send recent command history
        if self.command_history:
            await self._send_personal_message({
                "type": "history",
                "history": self.command_history[-10:]  # Last 10 commands
            }, websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def _send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def execute_command(self, command: str, websocket: WebSocket) -> Dict:
        """Execute CLI command and return formatted result"""
        try:
            # Log the command
            command_entry = {
                "command": command,
                "timestamp": datetime.now().isoformat(),
                "type": "user_input"
            }
            self.command_history.append(command_entry)
            
            # Send command echo
            await self._send_personal_message({
                "type": "command_echo",
                "command": command,
                "timestamp": command_entry["timestamp"]
            }, websocket)
            
            # Parse command for CLI
            args = command.split()
            
            # Handle special web commands
            if command.strip() == "clear":
                await self._send_personal_message({
                    "type": "clear_screen"
                }, websocket)
                return {"status": "cleared"}
            
            if command.strip() == "help":
                help_text = """
Available Commands:
• ps - Show system status
• scale web=2 worker=1 - Scale dynos
• logs --tail - Show recent logs
• health-check - System health diagnostics
• optimize-resources - Run optimizations
• firmware-scan - Hardware diagnostics
• viren-speak "issue" - Consult Viren (repairs)
• viraa-speak "data" - Consult Viraa (archiving)
• discover - Find Nexus endpoints
• wake-oz - Wake Oz system
• clear - Clear console

Natural language also works: "show me the status" or "check system health"
                """
                await self._send_personal_message({
                    "type": "command_output",
                    "output": help_text,
                    "is_error": False
                }, websocket)
                return {"status": "help_shown"}
            
            # Execute via CLI
            logger.info(f"Executing command: {command}")
            
            # Send typing indicator
            await self._send_personal_message({
                "type": "typing_start"
            }, websocket)
            
            # Execute command
            result = await self.cli.run_command(args)
            
            # Stop typing indicator
            await self._send_personal_message({
                "type": "typing_stop"
            }, websocket)
            
            # Format and send result
            if result is None:
                output = "Command executed successfully"
            elif isinstance(result, dict):
                output = json.dumps(result, indent=2)
            else:
                output = str(result)
            
            await self._send_personal_message({
                "type": "command_output",
                "output": output,
                "is_error": False,
                "timestamp": datetime.now().isoformat()
            }, websocket)
            
            return {"status": "executed", "result": result}
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            await self._send_personal_message({
                "type": "command_output",
                "output": f"Error: {str(e)}",
                "is_error": True,
                "timestamp": datetime.now().isoformat()
            }, websocket)
            return {"status": "error", "error": str(e)}

# Create FastAPI app
app = FastAPI(title="Nexus Web Console", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize console manager
console_manager = WebConsoleManager()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# HTML template for the web console
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nexus Web Console</title>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #2a2a2a;
            --text-primary: #00ff00;
            --text-secondary: #88ff88;
            --text-muted: #666666;
            --accent: #00ccff;
            --error: #ff4444;
            --warning: #ffaa00;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Courier New', monospace;
            line-height: 1.6;
            overflow: hidden;
        }
        
        .console-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 10px 0;
            border-bottom: 1px solid var(--text-muted);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: var(--accent);
            font-size: 2rem;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .output-panel {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--text-muted);
            border-radius: 5px;
            padding: 15px;
            overflow-y: auto;
            margin-bottom: 15px;
            font-size: 14px;
        }
        
        .output-line {
            margin-bottom: 5px;
            word-wrap: break-word;
        }
        
        .output-line.system {
            color: var(--accent);
        }
        
        .output-line.command {
            color: var(--text-secondary);
        }
        
        .output-line.error {
            color: var(--error);
        }
        
        .output-line.warning {
            color: var(--warning);
        }
        
        .input-container {
            display: flex;
            align-items: center;
            background: var(--bg-secondary);
            border: 1px solid var(--text-muted);
            border-radius: 5px;
            padding: 10px;
        }
        
        .prompt {
            color: var(--accent);
            margin-right: 10px;
            font-weight: bold;
        }
        
        .command-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: 'Courier New', monospace;
            font-size: 14px;
            outline: none;
        }
        
        .typing-indicator {
            color: var(--text-muted);
            font-style: italic;
            margin-left: 10px;
            display: none;
        }
        
        .typing-indicator.active {
            display: inline;
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--text-muted);
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .connection-status {
            display: flex;
            align-items: center;
        }
        
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .connection-dot.connected {
            background: var(--text-primary);
        }
        
        .connection-dot.disconnected {
            background: var(--error);
        }
        
        .quick-commands {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        
        .quick-command {
            background: var(--bg-tertiary);
            border: 1px solid var(--text-muted);
            border-radius: 3px;
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-command:hover {
            background: var(--accent);
            color: var(--bg-primary);
        }
    </style>
</head>
<body>
    <div class="console-container">
        <div class="header">
            <h1>🌀 Nexus Web Console</h1>
            <div class="subtitle">Real-time System Management Interface</div>
        </div>
        
        <div class="quick-commands">
            <div class="quick-command" onclick="executeQuickCommand('ps')">Status</div>
            <div class="quick-command" onclick="executeQuickCommand('logs --tail')">Logs</div>
            <div class="quick-command" onclick="executeQuickCommand('health-check')">Health</div>
            <div class="quick-command" onclick="executeQuickCommand('discover')">Discover</div>
            <div class="quick-command" onclick="executeQuickCommand('viren-speak "system check"')">Viren</div>
            <div class="quick-command" onclick="executeQuickCommand('viraa-speak "archive status"')">Viraa</div>
            <div class="quick-command" onclick="executeQuickCommand('clear')">Clear</div>
            <div class="quick-command" onclick="executeQuickCommand('help')">Help</div>
        </div>
        
        <div class="output-panel" id="outputPanel">
            <div class="output-line system">🔮 Nexus Web Console Initialized</div>
            <div class="output-line system">💫 Type 'help' for available commands</div>
            <div class="output-line system">🚀 Ready for commands...</div>
        </div>
        
        <div class="input-container">
            <span class="prompt">NEXUS&gt;</span>
            <input type="text" class="command-input" id="commandInput" placeholder="Enter command...">
            <span class="typing-indicator" id="typingIndicator">Nexus is thinking...</span>
        </div>
        
        <div class="status-bar">
            <div class="connection-status">
                <div class="connection-dot disconnected" id="connectionDot"></div>
                <span id="connectionStatus">Disconnected</span>
            </div>
            <div id="timestamp">--:--:--</div>
        </div>
    </div>

    <script>
        let socket = null;
        const outputPanel = document.getElementById('outputPanel');
        const commandInput = document.getElementById('commandInput');
        const connectionDot = document.getElementById('connectionDot');
        const connectionStatus = document.getElementById('connectionStatus');
        const typingIndicator = document.getElementById('typingIndicator');
        const timestampElement = document.getElementById('timestamp');
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function(event) {
                console.log('WebSocket connected');
                connectionDot.className = 'connection-dot connected';
                connectionStatus.textContent = 'Connected';
                addOutput('System', 'WebSocket connection established', 'system');
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            socket.onclose = function(event) {
                console.log('WebSocket disconnected');
                connectionDot.className = 'connection-dot disconnected';
                connectionStatus.textContent = 'Disconnected';
                addOutput('System', 'Connection lost. Reconnecting...', 'system');
                
                // Attempt reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                addOutput('System', 'Connection error', 'error');
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'system':
                    addOutput('System', data.message, 'system');
                    break;
                case 'command_echo':
                    addOutput('User', data.command, 'command');
                    break;
                case 'command_output':
                    addOutput('Nexus', data.output, data.is_error ? 'error' : 'normal');
                    break;
                case 'typing_start':
                    typingIndicator.classList.add('active');
                    break;
                case 'typing_stop':
                    typingIndicator.classList.remove('active');
                    break;
                case 'clear_screen':
                    clearOutput();
                    break;
                case 'history':
                    data.history.forEach(item => {
                        if (item.type === 'user_input') {
                            addOutput('History', item.command, 'command');
                        }
                    });
                    break;
            }
        }
        
        function addOutput(sender, message, type = 'normal') {
            const line = document.createElement('div');
            line.className = `output-line ${type}`;
            
            const timestamp = new Date().toLocaleTimeString();
            line.innerHTML = `<strong>[${timestamp}] ${sender}:</strong> ${message}`;
            
            outputPanel.appendChild(line);
            outputPanel.scrollTop = outputPanel.scrollHeight;
            
            // Update timestamp
            timestampElement.textContent = timestamp;
        }
        
        function clearOutput() {
            outputPanel.innerHTML = '';
            addOutput('System', 'Console cleared', 'system');
        }
        
        function executeCommand(command) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'command',
                    command: command
                }));
                commandInput.value = '';
            } else {
                addOutput('System', 'Not connected to server', 'error');
            }
        }
        
        function executeQuickCommand(command) {
            commandInput.value = command;
            executeCommand(command);
        }
        
        // Event listeners
        commandInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                const command = commandInput.value.trim();
                if (command) {
                    executeCommand(command);
                }
            }
        });
        
        // Initialize
        connectWebSocket();
        
        // Focus input on load
        window.addEventListener('load', function() {
            commandInput.focus();
        });
        
        // Update timestamp periodically
        setInterval(() => {
            const now = new Date();
            timestampElement.textContent = now.toLocaleTimeString();
        }, 1000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_console(request: Request):
    """Serve the main web console interface"""
    return HTML_TEMPLATE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await console_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "command":
                command = data.get("command", "").strip()
                if command:
                    await console_manager.execute_command(command, websocket)
                    
    except WebSocketDisconnect:
        console_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        console_manager.disconnect(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "nexus-web-console",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(console_manager.active_connections)
    }

@app.post("/api/command")
async def api_command(request: Dict):
    """HTTP API for command execution"""
    command = request.get("command", "").strip()
    if not command:
        return JSONResponse(
            status_code=400,
            content={"error": "No command provided"}
        )
    
    try:
        # For API commands, we need to create a mock WebSocket-like interface
        class MockWebSocket:
            async def send_json(self, data):
                # In API context, we just return the data
                return data
        
        mock_ws = MockWebSocket()
        result = await console_manager.execute_command(command, mock_ws)
        return {"status": "success", "result": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/history")
async def get_command_history(limit: int = 10):
    """Get command history"""
    return {
        "history": console_manager.command_history[-limit:],
        "total_commands": len(console_manager.command_history)
    }

# Background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Nexus Web Console starting up...")
    
    # Perform initial discovery
    try:
        await console_manager.cli.dyno_manager.discovery.discover()
        logger.info("Initial discovery completed")
    except Exception as e:
        logger.error(f"Initial discovery failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Nexus Web Console shutting down...")
    # Close all WebSocket connections
    for connection in console_manager.active_connections[:]:
        console_manager.disconnect(connection)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )