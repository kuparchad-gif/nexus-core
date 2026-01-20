#!/usr/bin/env python3
"""
Cloud Viren UI - Modal Web Interface
Provides a web interface for Cloud Viren running on Modal
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CloudVirenUI")

try:
    import gradio as gr
    from gradio.themes.base import Base
    from gradio.themes.utils import colors, sizes
except ImportError:
    logger.error("Gradio not installed. Install with: pip install gradio>=3.50.2")
    sys.exit(1)

# Import auth and models
from auth import Auth
from login_ui import create_login_ui
from models import ModelManager

# Theme colors
THEME_COLORS = {
    "primary": "#A2799A",    # Rich purple
    "secondary": "#93AEC5",  # Medium blue
    "tertiary": "#AFC5DC",   # Light blue
    "surface": "#C6D6E2",    # Very light blue
    "background": "#D8E3EB", # Pale blue
    "text": "#EBF2F6"        # Off-white
}

# Custom theme
class GlassTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.Color(
                c50="#f5f3ff",
                c100="#ede9fe",
                c200="#ddd6fe",
                c300="#c4b5fd",
                c400="#a78bfa",
                c500=THEME_COLORS["primary"],
                c600="#7c3aed",
                c700="#6d28d9",
                c800="#5b21b6",
                c900="#4c1d95",
                c950="#2e1065",
            ),
            secondary_hue=colors.Color(
                c50="#f0f9ff",
                c100="#e0f2fe",
                c200="#bae6fd",
                c300="#7dd3fc",
                c400="#38bdf8",
                c500=THEME_COLORS["secondary"],
                c600="#0284c7",
                c700="#0369a1",
                c800="#075985",
                c900="#0c4a6e",
                c950="#082f49",
            ),
            neutral_hue=colors.Color(
                c50="#f8fafc",
                c100="#f1f5f9",
                c200="#e2e8f0",
                c300="#cbd5e1",
                c400="#94a3b8",
                c500="#64748b",
                c600="#475569",
                c700="#334155",
                c800="#1e293b",
                c900="#0f172a",
                c950="#020617",
            ),
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_lg,
            text_size=sizes.text_md,
        )

class CloudVirenUI:
    """
    Cloud Viren UI
    Provides a web interface for Cloud Viren running on Modal
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        """Initialize the Cloud Viren UI"""
        self.api_url = api_url or os.environ.get("VIREN_API_URL", "https://viren-cloud--viren-api.modal.run")
        self.api_key = api_key or os.environ.get("VIREN_API_KEY", "")
        self.auth = Auth()
        self.model_manager = ModelManager()
        self.interface = None
        
        logger.info("Cloud Viren UI initialized")
    
    def start(self, port: int = 7860, share: bool = False, inbrowser: bool = True) -> None:
        """Start the Cloud Viren UI"""
        logger.info(f"Starting Cloud Viren UI on port {port}")
        
        # Create login UI
        login_ui = create_login_ui(self._create_main_ui)
        
        # Create main UI
        main_ui = self._create_main_ui()
        
        # Create auth wrapper
        with gr.Blocks(theme=GlassTheme(), css=self._get_css()) as interface:
            # Orb video background
            gr.HTML("""
            <div id='orb-background'>
                <video id='orb-video' autoplay loop muted playsinline style="position: fixed; width: 100vw; height: 100vh; object-fit: cover; z-index: -2;">
                    <source src="https://storage.googleapis.com/viren-assets/morph_orb.mp4" type="video/mp4">
                </video>
            </div>
            """)
            
            # Check for token in localStorage
            gr.HTML("""
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                const token = localStorage.getItem('viren_token');
                if (token) {
                    // Validate token
                    fetch('/validate_token', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({token: token})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.valid) {
                            localStorage.removeItem('viren_token');
                            document.getElementById('login-container').style.display = 'flex';
                            document.getElementById('main-container').style.display = 'none';
                        } else {
                            document.getElementById('login-container').style.display = 'none';
                            document.getElementById('main-container').style.display = 'flex';
                            document.getElementById('user-display').innerText = data.username;
                        }
                    });
                } else {
                    document.getElementById('login-container').style.display = 'flex';
                    document.getElementById('main-container').style.display = 'none';
                }
            });
            </script>
            """)
            
            # Login container
            with gr.Column(elem_id="login-container"):
                login_ui.render()
            
            # Main container
            with gr.Column(elem_id="main-container", visible=False):
                main_ui.render()
        
        # Add API endpoints
        @interface.load(api_name="validate_token")
        def validate_token(token):
            valid, session = self.auth.validate_token(token)
            if valid:
                return {"valid": True, "username": session["username"], "role": session["role"]}
            else:
                return {"valid": False}
        
        # Start interface
        self.interface = interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=inbrowser
        )
    
    def _create_main_ui(self):
        """Create the main UI"""
        with gr.Blocks() as main_ui:
            with gr.Row(elem_id="header"):
                gr.Markdown("# Cloud Viren", elem_id="title")
                with gr.Column(elem_id="user-info"):
                    gr.HTML("<div>Logged in as: <span id='user-display'></span></div>")
                    logout_btn = gr.Button("Logout", elem_id="logout-btn")
            
            with gr.Tabs() as tabs:
                with gr.Tab("Chat", id="chat-tab"):
                    self._create_chat_tab()
                
                with gr.Tab("Models", id="models-tab"):
                    self._create_models_tab()
                
                with gr.Tab("Weaviate", id="weaviate-tab"):
                    self._create_weaviate_tab()
                
                with gr.Tab("Binary Protocol", id="binary-tab"):
                    self._create_binary_tab()
                
                with gr.Tab("System", id="system-tab"):
                    self._create_system_tab()
            
            # Logout handler
            logout_btn.click(
                fn=lambda: None,
                inputs=[],
                outputs=[],
                _js="() => { localStorage.removeItem('viren_token'); window.location.reload(); }"
            )
        
        return main_ui
    
    def _create_chat_tab(self):
        """Create the chat tab"""
        with gr.Column(elem_id="chat-area"):
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot",
                show_copy_button=True,
                layout="panel",
                height=500
            )
            
            with gr.Row():
                with gr.Column(scale=4):
                    chat_input = gr.Textbox(
                        placeholder="Enter your query...",
                        show_label=False,
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=[m["name"] for m in self.model_manager.get_available_models()],
                            value=self.model_manager.get_available_models()[0]["name"] if self.model_manager.get_available_models() else None
                        )
                        
                        send_btn = gr.Button("Send", variant="primary")
        
        # Chat handlers
        def user_message(message, history):
            return "", history + [[message, None]]
        
        def bot_response(history, model_name):
            user_message = history[-1][0]
            response = self._query_api(user_message, model_name)
            
            # Simulate typing effect
            history[-1][1] = ""
            for char in response:
                history[-1][1] += char
                time.sleep(0.01)
                yield history
        
        send_btn.click(user_message, [chat_input, chatbot], [chat_input, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown], chatbot
        )
        
        chat_input.submit(user_message, [chat_input, chatbot], [chat_input, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown], chatbot
        )
    
    def _create_models_tab(self):
        """Create the models tab"""
        with gr.Column(elem_id="models-area"):
            gr.Markdown("## AI Models")
            
            with gr.Row():
                with gr.Column(scale=2):
                    models_json = gr.JSON(
                        label="Available Models",
                        value=self.model_manager.get_available_models()
                    )
                    
                    refresh_models_btn = gr.Button("Refresh Models")
                
                with gr.Column(scale=3):
                    with gr.Accordion("Add/Edit Model", open=True):
                        model_id = gr.Textbox(label="Model ID", placeholder="e.g., mistral")
                        model_name = gr.Textbox(label="Model Name", placeholder="e.g., Mistral-7B")
                        model_size = gr.Textbox(label="Model Size", placeholder="e.g., 7B")
                        model_endpoint = gr.Textbox(label="Model Endpoint", placeholder="e.g., mistralai/Mistral-7B-Instruct-v0.2")
                        model_context = gr.Number(label="Context Length", value=2048, precision=0)
                        model_enabled = gr.Checkbox(label="Enabled", value=True)
                        
                        with gr.Row():
                            add_model_btn = gr.Button("Add Model")
                            update_model_btn = gr.Button("Update Model")
                            delete_model_btn = gr.Button("Delete Model")
                        
                        model_result = gr.Markdown()
        
        # Model handlers
        def refresh_models():
            return self.model_manager.get_available_models()
        
        def add_model(model_id, name, size, endpoint, context, enabled):
            if not model_id or not name or not size or not endpoint:
                return "**Error:** All fields are required"
            
            model_config = {
                "name": name,
                "size": size,
                "endpoint": endpoint,
                "context_length": int(context),
                "enabled": enabled
            }
            
            success = self.model_manager.add_model(model_id, model_config)
            if success:
                return "**Success:** Model added"
            else:
                return "**Error:** Failed to add model"
        
        def update_model(model_id, name, size, endpoint, context, enabled):
            if not model_id:
                return "**Error:** Model ID is required"
            
            model_config = {}
            if name:
                model_config["name"] = name
            if size:
                model_config["size"] = size
            if endpoint:
                model_config["endpoint"] = endpoint
            if context:
                model_config["context_length"] = int(context)
            model_config["enabled"] = enabled
            
            success = self.model_manager.update_model(model_id, model_config)
            if success:
                return "**Success:** Model updated"
            else:
                return "**Error:** Failed to update model"
        
        # Connect handlers
        refresh_models_btn.click(refresh_models, [], models_json)
        add_model_btn.click(add_model, [model_id, model_name, model_size, model_endpoint, model_context, model_enabled], model_result)
        update_model_btn.click(update_model, [model_id, model_name, model_size, model_endpoint, model_context, model_enabled], model_result)
    
    def _create_weaviate_tab(self):
        """Create the Weaviate tab"""
        with gr.Column(elem_id="weaviate-area"):
            gr.Markdown("## Weaviate Vector Database")
            
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Vector Query",
                        placeholder="Enter vector search query...",
                        lines=3
                    )
                    
                    with gr.Row():
                        class_dropdown = gr.Dropdown(
                            label="Class",
                            choices=["TechnicalKnowledge", "ProblemSolvingConcept", "TroubleshootingTool", "BinaryMemoryShard"],
                            value="TechnicalKnowledge"
                        )
                        
                        limit_slider = gr.Slider(
                            label="Limit",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                    
                    search_btn = gr.Button("Search", variant="primary")
                
                with gr.Column(scale=4):
                    results_json = gr.JSON(label="Search Results")
        
        # Weaviate handlers
        def search_weaviate(query, class_name, limit):
            try:
                result = self._search_weaviate(query, class_name, limit)
                return result
            except Exception as e:
                return {"error": str(e)}
        
        # Connect handlers
        search_btn.click(search_weaviate, [query_input, class_dropdown, limit_slider], results_json)
    
    def _create_binary_tab(self):
        """Create the Binary Protocol tab"""
        with gr.Column(elem_id="binary-area"):
            gr.Markdown("## Binary Protocol")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Memory Shards", open=True):
                        refresh_shards_btn = gr.Button("Refresh Shards")
                        shard_stats = gr.JSON(label="Shard Statistics")
                
                with gr.Column(scale=1):
                    with gr.Accordion("Protocol Status", open=True):
                        refresh_status_btn = gr.Button("Check Status")
                        protocol_status = gr.JSON(label="Protocol Status")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Memory Operations")
                    
                    operation_dropdown = gr.Dropdown(
                        label="Operation",
                        choices=["Initialize", "Defragment", "Backup", "Restore"],
                        value="Initialize"
                    )
                    
                    execute_btn = gr.Button("Execute", variant="primary")
                    operation_result = gr.Textbox(label="Result", lines=3)
        
        # Binary Protocol handlers
        def get_shard_stats():
            try:
                return self._get_shard_stats()
            except Exception as e:
                return {"error": str(e)}
        
        def get_protocol_status():
            try:
                return self._get_protocol_status()
            except Exception as e:
                return {"error": str(e)}
        
        def execute_operation(operation):
            try:
                return self._execute_operation(operation)
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Connect handlers
        refresh_shards_btn.click(get_shard_stats, [], shard_stats)
        refresh_status_btn.click(get_protocol_status, [], protocol_status)
        execute_btn.click(execute_operation, [operation_dropdown], operation_result)
    
    def _create_system_tab(self):
        """Create the System tab"""
        with gr.Column(elem_id="system-area"):
            gr.Markdown("## System Status")
            
            with gr.Row():
                with gr.Column(scale=1):
                    check_system_btn = gr.Button("Check System Status")
                    system_status = gr.JSON(label="System Status")
                
                with gr.Column(scale=1):
                    logs_dropdown = gr.Dropdown(
                        label="Log Type",
                        choices=["System", "Weaviate", "Binary Protocol", "API"],
                        value="System"
                    )
                    
                    view_logs_btn = gr.Button("View Logs")
                    logs_output = gr.Textbox(label="Logs", lines=10)
        
        # System handlers
        def get_system_status():
            try:
                return self._get_system_status()
            except Exception as e:
                return {"error": str(e)}
        
        def get_logs(log_type):
            try:
                return self._get_logs(log_type)
            except Exception as e:
                return f"Error retrieving logs: {str(e)}"
        
        # Connect handlers
        check_system_btn.click(get_system_status, [], system_status)
        view_logs_btn.click(get_logs, [logs_dropdown], logs_output)
    
    def _get_css(self) -> str:
        """Get CSS for the UI"""
        return """
        /* Base styles */
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        
        /* Main container */
        #main-container {
            display: none;
            height: 100vh;
            width: 100vw;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Login container */
        #login-container {
            display: flex;
            height: 100vh;
            width: 100vw;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        
        /* Header */
        #header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            margin: 10px;
        }
        
        #title {
            color: white;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            margin: 0;
        }
        
        #user-info {
            display: flex;
            align-items: center;
            color: white;
        }
        
        #logout-btn {
            background: linear-gradient(135deg, #A2799A, #93AEC5);
            border-radius: 15px;
            border: none;
            padding: 5px 15px;
            color: white;
            margin-left: 15px;
        }
        
        /* Transparent glass-like areas */
        #chat-area, #models-area, #weaviate-area, #binary-area, #system-area {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 30px;
            padding: 16px;
            backdrop-filter: blur(10px);
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
            z-index: 2;
            margin: 10px;
            height: calc(100vh - 150px);
            overflow-y: auto;
        }
        
        #chatbot {
            color: white;
            font-size: 1rem;
        }
        
        /* Round floating buttons */
        #voice-buttons {
            position: fixed;
            bottom: 24px;
            width: 100%;
            display: flex;
            justify-content: center;
            gap: 20px;
            z-index: 10;
        }
        
        .circle-button {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #C0C0C0, #B0E0E6, #D8BFD8, #ffffff);
            box-shadow: 0 0 20px rgba(200, 200, 255, 0.5);
            font-size: 1.5rem;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .circle-button:hover {
            transform: scale(1.1);
            box-shadow: 0 0 30px rgba(255, 255, 255, 0.7);
        }
        
        /* Orb video */
        #orb-video {
            opacity: 0.15;
            transition: opacity 0.5s ease;
        }
        
        /* When AI is speaking */
        #orb-video.speaking {
            opacity: 0.6;
            filter: saturate(150%) brightness(1.3);
            transition: all 0.3s ease-in-out;
        }
        """
    
    def _query_api(self, query: str, model_name: str = None) -> str:
        """Query the Cloud Viren API"""
        try:
            # Get model by name
            model_id = None
            if model_name:
                for model in self.model_manager.get_available_models():
                    if model["name"] == model_name:
                        model_id = model["id"]
                        break
            
            headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
            payload = {"query": query}
            
            if model_id:
                payload["model_id"] = model_id
            
            import requests
            response = requests.post(
                f"{self.api_url}/query",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("result", "No result returned from API")
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Cloud Viren API: {str(e)}"
    
    def _search_weaviate(self, query: str, class_name: str, limit: int) -> Dict[str, Any]:
        """Search Weaviate vector database"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
            response = requests.post(
                f"{self.api_url}/weaviate/search",
                headers=headers,
                json={"query": query, "class": class_name, "limit": limit},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Error searching Weaviate: {str(e)}"}
    
    def _get_shard_stats(self) -> Dict[str, Any]:
        """Get Binary Protocol shard statistics"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key}
            response = requests.get(
                f"{self.api_url}/binary/shards",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Error getting shard stats: {str(e)}"}
    
    def _get_protocol_status(self) -> Dict[str, Any]:
        """Get Binary Protocol status"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key}
            response = requests.get(
                f"{self.api_url}/binary/status",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Error getting protocol status: {str(e)}"}
    
    def _execute_operation(self, operation: str) -> str:
        """Execute Binary Protocol operation"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
            response = requests.post(
                f"{self.api_url}/binary/execute",
                headers=headers,
                json={"operation": operation.lower()},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", "Operation completed")
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error executing operation: {str(e)}"}
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key}
            response = requests.get(
                f"{self.api_url}/system/status",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Error getting system status: {str(e)}"}
    
    def _get_logs(self, log_type: str) -> str:
        """Get system logs"""
        try:
            import requests
            headers = {"X-API-Key": self.api_key}
            response = requests.get(
                f"{self.api_url}/system/logs/{log_type.lower()}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("logs", "No logs available")
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error getting logs: {str(e)}"