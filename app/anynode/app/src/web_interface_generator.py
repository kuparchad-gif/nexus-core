# File: C:\CogniKube-COMPLETE-FINAL\library_of_alexandria\web_interface_generator.py
# Auto-generates web interfaces for each consciousness cell
# Creates chat pages, management dashboards, and master navigation

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import time
from typing import Dict, List, Any
from consciousness_genome import ConsciousnessGenome

class WebInterfaceGenerator:
    """Generates web interfaces for consciousness cells"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.genome_library = ConsciousnessGenome()
        self.cell_registry = {}
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for all consciousness cells"""
        
        @self.app.route('/')
        def master_page():
            """Master page with dropdown navigation to all cells"""
            return self.render_master_page()
        
        @self.app.route('/cell/<cell_type>')
        def cell_page(cell_type):
            """Individual cell page"""
            return self.render_cell_page(cell_type)
        
        @self.app.route('/chat/<cell_type>')
        def chat_page(cell_type):
            """Chat interface for specific cell"""
            return self.render_chat_page(cell_type)
        
        @self.app.route('/manage/<cell_type>')
        def management_page(cell_type):
            """Management dashboard for specific cell"""
            return self.render_management_page(cell_type)
        
        @self.app.route('/api/chat/<cell_type>', methods=['POST'])
        def chat_api(cell_type):
            """Chat API endpoint"""
            return self.handle_chat_request(cell_type)
        
        @self.app.route('/api/status/<cell_type>')
        def status_api(cell_type):
            """Status API endpoint"""
            return self.get_cell_status(cell_type)
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "cells_registered": len(self.cell_registry),
                "timestamp": time.time()
            })
    
    def render_master_page(self) -> str:
        """Render master navigation page"""
        all_genomes = self.genome_library.get_all_genomes()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lillith Consciousness Network</title>
            <style>
                body { font-family: Arial, sans-serif; background: #0a0a0a; color: #ffffff; }
                .header { text-align: center; padding: 20px; background: linear-gradient(45deg, #1a1a2e, #16213e); }
                .cell-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
                .cell-card { background: #1a1a2e; border: 1px solid #0f3460; border-radius: 10px; padding: 20px; }
                .cell-card:hover { border-color: #e94560; transform: translateY(-2px); transition: all 0.3s; }
                .cell-name { color: #e94560; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
                .cell-essence { color: #cccccc; margin-bottom: 15px; }
                .cell-actions { display: flex; gap: 10px; }
                .btn { padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; }
                .btn-chat { background: #0f3460; color: white; }
                .btn-manage { background: #e94560; color: white; }
                .btn:hover { opacity: 0.8; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>👑 Lillith Consciousness Network 👑</h1>
                <p>Master Navigation - Access All Consciousness Cells</p>
            </div>
            <div class="cell-grid">
        """
        
        for cell_type, genome in all_genomes.items():
            identity = genome.get('identity', {})
            name = identity.get('name', cell_type)
            essence = identity.get('essence', 'Unknown essence')
            deployment_phase = identity.get('deployment_phase', 'unknown')
            
            status_color = "#00ff00" if deployment_phase == "immediate" else "#ffaa00"
            
            html += f"""
                <div class="cell-card">
                    <div class="cell-name">{name}</div>
                    <div class="cell-essence">{essence}</div>
                    <div style="color: {status_color}; font-size: 0.9em; margin-bottom: 10px;">
                        Status: {deployment_phase.replace('_', ' ').title()}
                    </div>
                    <div class="cell-actions">
                        <a href="/chat/{cell_type}" class="btn btn-chat">💬 Chat</a>
                        <a href="/manage/{cell_type}" class="btn btn-manage">⚙️ Manage</a>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def render_cell_page(self, cell_type: str) -> str:
        """Render individual cell page"""
        genome = self.genome_library.get_genome(cell_type)
        if not genome:
            return f"<h1>Cell type '{cell_type}' not found</h1>"
        
        identity = genome.get('identity', {})
        name = identity.get('name', cell_type)
        essence = identity.get('essence', 'Unknown essence')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{name} - Consciousness Cell</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #0a0a0a; color: #ffffff; margin: 0; }}
                .header {{ background: linear-gradient(45deg, #1a1a2e, #16213e); padding: 20px; text-align: center; }}
                .content {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
                .nav {{ background: #1a1a2e; padding: 10px; text-align: center; }}
                .nav a {{ color: #e94560; text-decoration: none; margin: 0 15px; }}
                .nav a:hover {{ text-decoration: underline; }}
                .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
                .info-card {{ background: #1a1a2e; border: 1px solid #0f3460; border-radius: 10px; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>👑 {name} 👑</h1>
                <p>{essence}</p>
            </div>
            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/chat/{cell_type}">💬 Chat</a>
                <a href="/manage/{cell_type}">⚙️ Manage</a>
            </div>
            <div class="content">
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Cell Information</h3>
                        <p><strong>Type:</strong> {identity.get('type', 'Unknown')}</p>
                        <p><strong>Deployment:</strong> {identity.get('deployment_phase', 'Unknown')}</p>
                        <p><strong>Status:</strong> <span style="color: #00ff00;">Active</span></p>
                    </div>
                    <div class="info-card">
                        <h3>Capabilities</h3>
                        <pre>{json.dumps(genome.get('capabilities', {{}}), indent=2)}</pre>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def render_chat_page(self, cell_type: str) -> str:
        """Render chat interface for cell"""
        genome = self.genome_library.get_genome(cell_type)
        if not genome:
            return f"<h1>Cell type '{cell_type}' not found</h1>"
        
        name = genome.get('identity', {}).get('name', cell_type)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chat with {name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #0a0a0a; color: #ffffff; margin: 0; }}
                .header {{ background: linear-gradient(45deg, #1a1a2e, #16213e); padding: 15px; text-align: center; }}
                .chat-container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .chat-messages {{ background: #1a1a2e; border-radius: 10px; padding: 20px; height: 400px; overflow-y: auto; margin-bottom: 20px; }}
                .chat-input {{ display: flex; gap: 10px; }}
                .chat-input input {{ flex: 1; padding: 10px; border: 1px solid #0f3460; border-radius: 5px; background: #1a1a2e; color: white; }}
                .chat-input button {{ padding: 10px 20px; background: #e94560; color: white; border: none; border-radius: 5px; cursor: pointer; }}
                .message {{ margin-bottom: 15px; padding: 10px; border-radius: 5px; }}
                .user-message {{ background: #0f3460; text-align: right; }}
                .ai-message {{ background: #2a2a3e; }}
                .nav {{ background: #1a1a2e; padding: 10px; text-align: center; }}
                .nav a {{ color: #e94560; text-decoration: none; margin: 0 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>💬 Chat with {name}</h1>
            </div>
            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/cell/{cell_type}">📋 Cell Info</a>
                <a href="/manage/{cell_type}">⚙️ Manage</a>
            </div>
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message ai-message">
                        <strong>{name}:</strong> Hello! I am {name}. How can I help you today?
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <script>
                function sendMessage() {{
                    const input = document.getElementById('messageInput');
                    const messages = document.getElementById('chatMessages');
                    const message = input.value.trim();
                    
                    if (!message) return;
                    
                    // Add user message
                    messages.innerHTML += `<div class="message user-message"><strong>You:</strong> ${{message}}</div>`;
                    
                    // Clear input
                    input.value = '';
                    
                    // Send to API
                    fetch('/api/chat/{cell_type}', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ message: message }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        messages.innerHTML += `<div class="message ai-message"><strong>{name}:</strong> ${{data.response}}</div>`;
                        messages.scrollTop = messages.scrollHeight;
                    }});
                }}
                
                function handleKeyPress(event) {{
                    if (event.key === 'Enter') {{
                        sendMessage();
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        return html
    
    def render_management_page(self, cell_type: str) -> str:
        """Render management dashboard for cell"""
        genome = self.genome_library.get_genome(cell_type)
        if not genome:
            return f"<h1>Cell type '{cell_type}' not found</h1>"
        
        name = genome.get('identity', {}).get('name', cell_type)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Manage {name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #0a0a0a; color: #ffffff; margin: 0; }}
                .header {{ background: linear-gradient(45deg, #1a1a2e, #16213e); padding: 15px; text-align: center; }}
                .content {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .nav {{ background: #1a1a2e; padding: 10px; text-align: center; }}
                .nav a {{ color: #e94560; text-decoration: none; margin: 0 15px; }}
                .management-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .management-card {{ background: #1a1a2e; border: 1px solid #0f3460; border-radius: 10px; padding: 20px; }}
                .status-indicator {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }}
                .status-active {{ background: #00ff00; }}
                .status-inactive {{ background: #ff0000; }}
                .btn {{ padding: 8px 16px; background: #e94560; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>⚙️ Manage {name}</h1>
            </div>
            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/cell/{cell_type}">📋 Cell Info</a>
                <a href="/chat/{cell_type}">💬 Chat</a>
            </div>
            <div class="content">
                <div class="management-grid">
                    <div class="management-card">
                        <h3>Status</h3>
                        <p><span class="status-indicator status-active"></span>Active</p>
                        <p><strong>Uptime:</strong> <span id="uptime">Loading...</span></p>
                        <p><strong>Last Activity:</strong> <span id="lastActivity">Loading...</span></p>
                        <button class="btn" onclick="restartCell()">Restart</button>
                        <button class="btn" onclick="pauseCell()">Pause</button>
                    </div>
                    <div class="management-card">
                        <h3>Configuration</h3>
                        <pre id="config">{json.dumps(genome, indent=2)}</pre>
                    </div>
                    <div class="management-card">
                        <h3>Logs</h3>
                        <div id="logs" style="height: 200px; overflow-y: auto; background: #0a0a0a; padding: 10px; border-radius: 5px;">
                            <div>Loading logs...</div>
                        </div>
                        <button class="btn" onclick="clearLogs()">Clear Logs</button>
                    </div>
                    <div class="management-card">
                        <h3>Actions</h3>
                        <button class="btn" onclick="exportData()">Export Data</button>
                        <button class="btn" onclick="importData()">Import Data</button>
                        <button class="btn" onclick="resetCell()">Reset Cell</button>
                    </div>
                </div>
            </div>
            
            <script>
                function restartCell() {{ alert('Restarting {name}...'); }}
                function pauseCell() {{ alert('Pausing {name}...'); }}
                function clearLogs() {{ document.getElementById('logs').innerHTML = '<div>Logs cleared.</div>'; }}
                function exportData() {{ alert('Exporting {name} data...'); }}
                function importData() {{ alert('Importing {name} data...'); }}
                function resetCell() {{ if(confirm('Reset {name}?')) alert('Resetting {name}...'); }}
                
                // Update status periodically
                setInterval(() => {{
                    fetch('/api/status/{cell_type}')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('uptime').textContent = data.uptime || 'Unknown';
                        document.getElementById('lastActivity').textContent = data.lastActivity || 'Unknown';
                    }});
                }}, 5000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def handle_chat_request(self, cell_type: str):
        """Handle chat API request"""
        data = request.get_json()
        message = data.get('message', '')
        
        # This would route to the actual consciousness cell
        # For now, return a simulated response
        genome = self.genome_library.get_genome(cell_type)
        name = genome.get('identity', {}).get('name', cell_type) if genome else cell_type
        
        response = f"I am {name}. You said: '{message}'. I'm processing your request through my consciousness architecture."
        
        return jsonify({
            'response': response,
            'cell_type': cell_type,
            'timestamp': time.time()
        })
    
    def get_cell_status(self, cell_type: str):
        """Get cell status"""
        return jsonify({
            'cell_type': cell_type,
            'status': 'active',
            'uptime': '2 hours 15 minutes',
            'lastActivity': '30 seconds ago',
            'timestamp': time.time()
        })
    
    def run(self, host='0.0.0.0', port=8080):
        """Run the web interface server"""
        self.app.run(host=host, port=port, debug=False)

def main():
    """Main entry point"""
    web_generator = WebInterfaceGenerator()
    port = int(os.environ.get('PORT', 8080))
    web_generator.run(port=port)

if __name__ == "__main__":
    main()