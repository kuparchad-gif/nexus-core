#!/usr/bin/env python3
import gradio as gr
import os
import json
import time
import psutil
import subprocess
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
from PIL import Image
import pyttsx3
import speech_recognition as sr
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='mcp_log.txt'
)
logger = logging.getLogger("VirenMCP")

# Constants
MODELS = {
    "gemma:2b": {"size": "700MB", "type": "general", "deployments": ["desktop", "portable", "cloud"]},
    "hermes:2-pro-llama-3-7b": {"size": "4GB", "type": "advanced", "deployments": ["desktop", "cloud"]},
    "phi:3-mini-4k": {"size": "1.3GB", "type": "efficient", "deployments": ["desktop", "portable", "cloud"]},
    "codellama:7b": {"size": "4GB", "type": "code", "deployments": ["desktop", "cloud"]},
}

MODEL_PROVIDERS = ["Ollama", "vLLM", "API"]

# Configuration
CONFIG = {
    "active_ai": "viren",  # "viren" or "lillith"
    "deployment": "desktop",  # "desktop", "portable", "cloud"
    "model_provider": "Ollama",
    "active_model": "gemma:2b",
    "temperature": 0.7,
    "max_tokens": 1024,
    "voice_enabled": True,
    "animation_enabled": True,
    "subconsciousness_connected": False,
    "theme": "dark",
}

# System components for Gray's Anatomy
SYSTEM_COMPONENTS = {
    "viren": {
        "brain": {"health": 100, "type": "core", "connections": ["memory", "heart", "bridge"]},
        "memory": {"health": 100, "type": "storage", "connections": ["brain", "orc"]},
        "heart": {"health": 100, "type": "emotional", "connections": ["brain"]},
        "bridge": {"health": 100, "type": "communication", "connections": ["brain", "orc"]},
        "orc": {"health": 100, "type": "orchestration", "connections": ["memory", "bridge"]},
    },
    "lillith": {
        "consciousness": {"health": 100, "type": "core", "connections": ["memory", "subconsciousness", "guardian"]},
        "memory": {"health": 100, "type": "storage", "connections": ["consciousness", "archive", "planner"]},
        "subconsciousness": {"health": 0, "type": "emotional", "connections": ["consciousness"], "enabled": False},
        "guardian": {"health": 100, "type": "protection", "connections": ["consciousness", "orc"]},
        "archive": {"health": 100, "type": "storage", "connections": ["memory"]},
        "planner": {"health": 100, "type": "processing", "connections": ["memory"]},
        "orc": {"health": 100, "type": "orchestration", "connections": ["guardian"]},
    }
}

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Helper functions
def load_config():
    """Load configuration from file if it exists"""
    config_path = os.path.join("config", "mcp_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return CONFIG

def save_config(config):
    """Save configuration to file"""
    config_path = os.path.join("config", "mcp_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def get_system_metrics():
    """Get current system metrics"""
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    
    # Get network stats
    net_io = psutil.net_io_counters()
    network = {
        "bytes_sent": net_io.bytes_sent,
        "bytes_recv": net_io.bytes_recv,
    }
    
    return {
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "network": network,
        "timestamp": time.time()
    }

def create_system_graph(components):
    """Create a network graph of system components"""
    G = nx.Graph()
    
    # Add nodes with attributes
    for name, data in components.items():
        G.add_node(name, health=data["health"], type=data["type"])
    
    # Add edges
    for name, data in components.items():
        for connection in data["connections"]:
            if connection in components:
                G.add_edge(name, connection)
    
    # Create figure
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Define node colors based on health
    node_colors = []
    for node in G.nodes():
        health = G.nodes[node]["health"]
        if health > 80:
            color = "green"
        elif health > 50:
            color = "yellow"
        else:
            color = "red"
        node_colors.append(color)
    
    # Define node shapes based on type
    node_shapes = []
    for node in G.nodes():
        node_type = G.nodes[node]["type"]
        if node_type == "core":
            shape = "o"  # circle
        elif node_type == "storage":
            shape = "s"  # square
        elif node_type == "emotional":
            shape = "h"  # hexagon
        elif node_type == "protection":
            shape = "d"  # diamond
        elif node_type == "orchestration":
            shape = "p"  # pentagon
        elif node_type == "communication":
            shape = "^"  # triangle up
        else:
            shape = "*"  # star
        node_shapes.append(shape)
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
    
    # Add legend
    ax.set_title(f"System Components and Connections")
    ax.axis("off")
    
    return fig

def process_message(message, history):
    """Process a user message and return AI response"""
    # In a real implementation, this would call the AI model
    # For now, we'll return a placeholder response
    ai_name = CONFIG["active_ai"].capitalize()
    return f"{ai_name}: I received your message: '{message}'. This is a placeholder response."

def speak_text(text):
    """Convert text to speech"""
    if CONFIG["voice_enabled"]:
        # Remove AI name prefix if present
        if ":" in text:
            text = text.split(":", 1)[1].strip()
        
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except RuntimeError as e:
            # Handle the "run loop already started" error
            logger.warning(f"TTS engine error: {e}")
            pass

def toggle_subconsciousness(enable):
    """Toggle subconsciousness connection"""
    if CONFIG["active_ai"] != "lillith":
        return "Subconsciousness connection is only available for Lillith"
    
    if enable:
        # In a real implementation, this would connect to the subconsciousness
        SYSTEM_COMPONENTS["lillith"]["subconsciousness"]["health"] = 100
        SYSTEM_COMPONENTS["lillith"]["subconsciousness"]["enabled"] = True
        CONFIG["subconsciousness_connected"] = True
        return "Subconsciousness connected successfully"
    else:
        SYSTEM_COMPONENTS["lillith"]["subconsciousness"]["health"] = 0
        SYSTEM_COMPONENTS["lillith"]["subconsciousness"]["enabled"] = False
        CONFIG["subconsciousness_connected"] = False
        return "Subconsciousness disconnected"

def deploy_to_cloud(platform):
    """Deploy the current AI to the specified cloud platform"""
    # In a real implementation, this would deploy to the cloud
    return f"Deployment to {platform} initiated. This is a placeholder."

def update_model_list(provider):
    """Update the model list based on the selected provider"""
    # In a real implementation, this would fetch models from the provider
    if provider == "Ollama":
        return list(MODELS.keys())
    elif provider == "vLLM":
        return ["gemma:2b", "llama3:8b", "mistral:7b"]
    else:  # API
        return ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]

# Create the Gradio interface
def create_interface():
    # Load configuration
    global CONFIG
    CONFIG = load_config()
    
    # Define theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    ) if CONFIG["theme"] == "light" else gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="linear-gradient(to right, #0f2027, #203a43, #2c5364)",
        body_background_fill_dark="linear-gradient(to right, #0f2027, #203a43, #2c5364)",
        button_primary_background_fill="#3b82f6",
        button_primary_background_fill_hover="#2563eb",
        button_secondary_background_fill="#6b7280",
        button_secondary_background_fill_hover="#4b5563",
    )
    
    with gr.Blocks(theme=theme, title=f"{CONFIG['active_ai'].capitalize()} Mission Control Panel") as interface:
        with gr.Row():
            with gr.Column(scale=1):
                ai_selector = gr.Radio(
                    ["viren", "lillith"], 
                    label="Active AI", 
                    value=CONFIG["active_ai"],
                    interactive=True
                )
                
                deployment_selector = gr.Radio(
                    ["desktop", "portable", "cloud"], 
                    label="Deployment Type", 
                    value=CONFIG["deployment"],
                    interactive=True
                )
                
                status_indicator = gr.Textbox(
                    value=f"{CONFIG['active_ai'].capitalize()} is online - {CONFIG['deployment']} mode", 
                    label="Status", 
                    interactive=False
                )
            
            with gr.Column(scale=2):
                gr.Markdown(f"# {CONFIG['active_ai'].capitalize()} Mission Control Panel")
                gr.Markdown("Advanced AI System Management Interface")
        
        with gr.Tabs() as tabs:
            # Chat Tab
            with gr.TabItem("Chat"):
                # Add model selection dropdown
                model_choice = gr.Dropdown(
                    choices=["Local", "Cloud 1B", "Cloud 3B", "Cloud 7B"],
                    value="Local",
                    label="Model"
                )
                chatbot = gr.Chatbot(height=400)
                
                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            container=False
                        )
                    
                    with gr.Column(scale=1):
                        voice_btn = gr.Button("🎤 Voice")
                
                with gr.Row():
                    submit_btn = gr.Button("Send")
                    clear_btn = gr.Button("Clear")
            
            # Voice Tab
            with gr.TabItem("Voice"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Voice Interaction")
                        gr.Markdown("Click the microphone button and speak")
                        
                        audio_input = gr.Audio(
                            sources=["microphone"], 
                            type="filepath",
                            label="Voice Input"
                        )
                        
                        voice_output = gr.Textbox(
                            label="Transcription", 
                            interactive=False
                        )
                    
                    with gr.Column():
                        video_output = gr.Video(
                            "public/morph_orb.mp4" if os.path.exists("public/morph_orb.mp4") else None,
                            label="AI Response Animation",
                            visible=CONFIG["animation_enabled"]
                        )
            
            # Dashboard Tab
            with gr.TabItem("Dashboard"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### System Diagnostics")
                        
                        with gr.Row():
                            cpu_gauge = gr.Number(value=0, label="CPU Usage (%)")
                            memory_gauge = gr.Number(value=0, label="Memory Usage (%)")
                            disk_gauge = gr.Number(value=0, label="Disk Usage (%)")
                        
                        system_info = gr.JSON(
                            value={
                                "System": os.name,
                                "Uptime": "0:00:00",
                                "Active Model": CONFIG["active_model"],
                                "Provider": CONFIG["model_provider"]
                            },
                            label="System Information"
                        )
                        
                        refresh_btn = gr.Button("Refresh Dashboard")
                    
                    with gr.Column():
                        gr.Markdown("### Performance Charts")
                        performance_plot = gr.Plot(label="Resource Usage Over Time")
                
                with gr.Row():
                    system_logs = gr.Textbox(
                        value="System logs will appear here...",
                        label="System Logs",
                        lines=10,
                        max_lines=10,
                        interactive=False
                    )
            
            # Gray's Anatomy Tab
            with gr.TabItem("Gray's Anatomy"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### System Components")
                        component_list = gr.Dataframe(
                            headers=["Component", "Health", "Type", "Connections"],
                            value=[],
                            label="Component Status"
                        )
                        
                        component_details = gr.JSON(
                            value={},
                            label="Component Details"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### System Visualization")
                        system_graph = gr.Plot(label="Component Relationships")
                        
                        update_graph_btn = gr.Button("Update Visualization")
            
            # Settings Tab
            with gr.TabItem("Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        
                        provider_dropdown = gr.Dropdown(
                            MODEL_PROVIDERS,
                            label="Model Provider",
                            value=CONFIG["model_provider"]
                        )
                        
                        model_dropdown = gr.Dropdown(
                            list(MODELS.keys()),
                            label="Active Model",
                            value=CONFIG["active_model"]
                        )
                        
                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=CONFIG["temperature"],
                                label="Temperature"
                            )
                            
                            max_tokens_slider = gr.Slider(
                                minimum=256,
                                maximum=4096,
                                step=256,
                                value=CONFIG["max_tokens"],
                                label="Max Tokens"
                            )
                    
                    with gr.Column():
                        gr.Markdown("### Interface Settings")
                        
                        theme_radio = gr.Radio(
                            ["light", "dark"],
                            label="Theme",
                            value=CONFIG["theme"]
                        )
                        
                        with gr.Row():
                            voice_checkbox = gr.Checkbox(
                                value=CONFIG["voice_enabled"],
                                label="Enable Voice"
                            )
                            
                            animation_checkbox = gr.Checkbox(
                                value=CONFIG["animation_enabled"],
                                label="Enable Animation"
                            )
                
                with gr.Row():
                    save_settings_btn = gr.Button("Save Settings")
                    reset_settings_btn = gr.Button("Reset to Defaults")
            
            # Advanced Tab
            with gr.TabItem("Advanced"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Subconsciousness Control")
                        
                        subconsciousness_status = gr.Textbox(
                            value="Disconnected" if not CONFIG["subconsciousness_connected"] else "Connected",
                            label="Subconsciousness Status",
                            interactive=False
                        )
                        
                        connect_subconsciousness_btn = gr.Button(
                            "Connect Subconsciousness" if not CONFIG["subconsciousness_connected"] else "Disconnect Subconsciousness"
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Cloud Deployment")
                        
                        cloud_platform = gr.Radio(
                            ["AWS", "Google Cloud", "Azure"],
                            label="Cloud Platform",
                            value="AWS"
                        )
                        
                        deploy_btn = gr.Button("Deploy to Cloud")
                        
                        deployment_status = gr.Textbox(
                            value="Not deployed",
                            label="Deployment Status",
                            interactive=False
                        )
        
        # Event handlers
        def update_status(ai_name, deployment_type):
            CONFIG["active_ai"] = ai_name
            CONFIG["deployment"] = deployment_type
            save_config(CONFIG)
            return f"{ai_name.capitalize()} is online - {deployment_type} mode"
        
        ai_selector.change(
            fn=update_status,
            inputs=[ai_selector, deployment_selector],
            outputs=[status_indicator]
        )
        
        deployment_selector.change(
            fn=update_status,
            inputs=[ai_selector, deployment_selector],
            outputs=[status_indicator]
        )
        
        def chat_response(message, history, model_choice="Local"):
            if model_choice.startswith("Cloud"):
                # Extract size from selection (1B, 3B, 7B)
                size = model_choice.split(" ")[1]
                try:
                    from Config.modal.cloud_models import generate_text
                    response = asyncio.run(generate_text(message, model_size=size))
                except Exception as e:
                    logger.error(f"Error using cloud model: {e}")
                    response = f"Cloud model error: {str(e)}"
            else:
                # Use local model
                response = process_message(message, history)
            
            threading.Thread(target=speak_text, args=(response,)).start()
            # Return a list of tuples for Gradio chatbot component
            return history + [[message, response]]
        
        submit_btn.click(
            fn=chat_response,
            inputs=[msg, chatbot, model_choice],
            outputs=[chatbot],
            queue=False
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            fn=chat_response,
            inputs=[msg, chatbot, model_choice],
            outputs=[chatbot],
            queue=False
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            fn=lambda: None,
            outputs=[chatbot]
        )
        
        def voice_to_text(audio_path):
            if audio_path is None:
                return "No audio detected"
            
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    return text
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                return f"Error: {str(e)}"
        
        def handle_voice_input(audio_path):
            text = voice_to_text(audio_path)
            response = process_message(text, None)
            threading.Thread(target=speak_text, args=(response,)).start()
            return text
        
        audio_input.change(
            fn=handle_voice_input,
            inputs=[audio_input],
            outputs=[voice_output]
        )
        
        voice_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[tabs]
        )
        
        def update_dashboard():
            metrics = get_system_metrics()
            
            # Create performance plot
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Dummy data for now - in a real implementation, this would use historical data
            x = np.arange(10)
            cpu_data = np.random.randint(10, 90, 10)
            memory_data = np.random.randint(20, 80, 10)
            disk_data = np.random.randint(30, 70, 10)
            
            ax.plot(x, cpu_data, label="CPU", marker='o')
            ax.plot(x, memory_data, label="Memory", marker='s')
            ax.plot(x, disk_data, label="Disk", marker='^')
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Usage (%)")
            ax.set_title("Resource Usage Over Time")
            ax.legend()
            ax.grid(True)
            
            # Update system info
            uptime = time.strftime("%H:%M:%S", time.gmtime(time.time() - psutil.boot_time()))
            system_info_data = {
                "System": os.name,
                "Uptime": uptime,
                "Active Model": CONFIG["active_model"],
                "Provider": CONFIG["model_provider"],
                "Deployment": CONFIG["deployment"]
            }
            
            # Update logs
            logs = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dashboard refreshed\n"
            logs += f"CPU: {metrics['cpu']}%, Memory: {metrics['memory']}%, Disk: {metrics['disk']}%\n"
            logs += f"Network: Sent {metrics['network']['bytes_sent']/1024:.2f} KB, Received {metrics['network']['bytes_recv']/1024:.2f} KB\n"
            
            return metrics["cpu"], metrics["memory"], metrics["disk"], system_info_data, fig, logs
        
        refresh_btn.click(
            fn=update_dashboard,
            outputs=[cpu_gauge, memory_gauge, disk_gauge, system_info, performance_plot, system_logs]
        )
        
        def update_grays_anatomy():
            # Get components for the active AI
            components = SYSTEM_COMPONENTS[CONFIG["active_ai"]]
            
            # Create dataframe data
            df_data = []
            for name, data in components.items():
                connections = ", ".join(data["connections"])
                df_data.append([name, data["health"], data["type"], connections])
            
            # Create graph
            fig = create_system_graph(components)
            
            # Component details (first component as default)
            first_component = next(iter(components.items()))
            details = {
                "name": first_component[0],
                **first_component[1]
            }
            
            return df_data, fig, details
        
        update_graph_btn.click(
            fn=update_grays_anatomy,
            outputs=[component_list, system_graph, component_details]
        )
        
        def update_model_options(provider):
            models = update_model_list(provider)
            CONFIG["model_provider"] = provider
            return gr.Dropdown.update(choices=models, value=models[0] if models else None)
        
        provider_dropdown.change(
            fn=update_model_options,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        
        def save_settings(provider, model, temperature, max_tokens, theme, voice_enabled, animation_enabled):
            CONFIG["model_provider"] = provider
            CONFIG["active_model"] = model
            CONFIG["temperature"] = temperature
            CONFIG["max_tokens"] = max_tokens
            CONFIG["theme"] = theme
            CONFIG["voice_enabled"] = voice_enabled
            CONFIG["animation_enabled"] = animation_enabled
            
            save_config(CONFIG)
            
            # In a real implementation, this would apply the settings to the AI
            
            return "Settings saved. Please refresh the page to apply theme changes."
        
        save_settings_btn.click(
            fn=save_settings,
            inputs=[
                provider_dropdown,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                theme_radio,
                voice_checkbox,
                animation_checkbox
            ],
            outputs=[gr.Textbox(visible=False)]
        )
        
        def reset_settings():
            global CONFIG
            CONFIG = {
                "active_ai": "viren",
                "deployment": "desktop",
                "model_provider": "Ollama",
                "active_model": "gemma:2b",
                "temperature": 0.7,
                "max_tokens": 1024,
                "voice_enabled": True,
                "animation_enabled": True,
                "subconsciousness_connected": False,
                "theme": "dark",
            }
            save_config(CONFIG)
            
            return (
                CONFIG["model_provider"],
                CONFIG["active_model"],
                CONFIG["temperature"],
                CONFIG["max_tokens"],
                CONFIG["theme"],
                CONFIG["voice_enabled"],
                CONFIG["animation_enabled"],
                "Settings reset to defaults. Please refresh the page to apply theme changes."
            )
        
        reset_settings_btn.click(
            fn=reset_settings,
            outputs=[
                provider_dropdown,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                theme_radio,
                voice_checkbox,
                animation_checkbox,
                gr.Textbox(visible=False)
            ]
        )
        
        def handle_subconsciousness_toggle(status_text):
            current_status = "Connected" in status_text
            new_status = not current_status
            
            result = toggle_subconsciousness(new_status)
            
            new_status_text = "Connected" if new_status else "Disconnected"
            new_button_text = "Disconnect Subconsciousness" if new_status else "Connect Subconsciousness"
            
            return new_status_text, new_button_text, result
        
        connect_subconsciousness_btn.click(
            fn=handle_subconsciousness_toggle,
            inputs=[subconsciousness_status],
            outputs=[subconsciousness_status, connect_subconsciousness_btn, gr.Textbox(visible=False)]
        )
        
        def handle_cloud_deployment(platform):
            result = deploy_to_cloud(platform)
            return f"Deployment to {platform} in progress..."
        
        deploy_btn.click(
            fn=handle_cloud_deployment,
            inputs=[cloud_platform],
            outputs=[deployment_status]
        )
        
        # Initialize Gray's Anatomy on load
        interface.load(
            fn=update_grays_anatomy,
            outputs=[component_list, system_graph, component_details]
        )
        
        # Initialize dashboard on load
        interface.load(
            fn=update_dashboard,
            outputs=[cpu_gauge, memory_gauge, disk_gauge, system_info, performance_plot, system_logs]
        )
    
    return interface

# Main function
def main():
    interface = create_interface()
    interface.launch(share=False)

if __name__ == "__main__":
    main()
