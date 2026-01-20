#!/usr/bin/env python3
"""
Viren Platinum Edition - Enhanced Mission Control Panel
"""

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
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
import webbrowser
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"viren_platinum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VirenPlatinum")

# Import existing modules
try:
    from model_service import ModelService
    from grays_anatomy import GraysAnatomy
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure you're running from the correct directory")
    sys.exit(1)

# Constants and configuration
CONFIG_DIR = os.path.join("config")
MODELS_CONFIG_PATH = os.path.join(CONFIG_DIR, "model_config.py")
SOULPRINT_PATH = os.path.join(CONFIG_DIR, "viren_soulprint.json")
MCP_CONFIG_PATH = os.path.join(CONFIG_DIR, "mcp_config.json")
ANIMATION_PATH = os.path.join("public", "morph_orb.mp4")

# Default configuration
DEFAULT_CONFIG = {
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

# Initialize services
model_service = None
anatomy_service = None
tts_engine = None

# Helper functions
def load_config():
    """Load configuration from file if it exists"""
    if os.path.exists(MCP_CONFIG_PATH):
        try:
            with open(MCP_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
    try:
        with open(MCP_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def load_soulprint():
    """Load soulprint configuration"""
    if os.path.exists(SOULPRINT_PATH):
        try:
            with open(SOULPRINT_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading soulprint: {e}")
    return {}

def initialize_services():
    """Initialize required services"""
    global model_service, anatomy_service, tts_engine
    
    # Initialize model service
    try:
        model_service = ModelService()
        logger.info("Model service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        model_service = None
    
    # Initialize Gray's Anatomy
    try:
        anatomy_service = GraysAnatomy("viren")
        logger.info("Gray's Anatomy service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Gray's Anatomy: {e}")
        anatomy_service = None
    
    # Initialize TTS engine
    try:
        tts_engine = pyttsx3.init()
        logger.info("TTS engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        tts_engine = None

def get_system_metrics():
    """Get current system metrics"""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
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
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            "cpu": 0,
            "memory": 0,
            "disk": 0,
            "network": {"bytes_sent": 0, "bytes_recv": 0},
            "timestamp": time.time()
        }

def process_message(message, history):
    """Process a user message and return AI response"""
    if not model_service:
        return "Error: Model service not initialized"
    
    try:
        # Get configuration
        config = load_config()
        
        # Set active model if needed
        if model_service.active_model != config["active_model"] or model_service.active_provider != config["model_provider"]:
            model_service.set_active_model(config["active_model"], config["model_provider"])
        
        # Generate response
        response = model_service.generate_text(
            message, 
            max_tokens=config["max_tokens"], 
            temperature=config["temperature"]
        )
        
        # Format response
        ai_name = config["active_ai"].capitalize()
        formatted_response = f"{ai_name}: {response}"
        
        # Log the interaction
        logger.info(f"User: {message}")
        logger.info(f"AI: {response}")
        
        return formatted_response
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"Error processing your message: {str(e)}"

def speak_text(text):
    """Convert text to speech"""
    if not tts_engine:
        logger.warning("TTS engine not initialized")
        return
    
    config = load_config()
    if not config["voice_enabled"]:
        return
    
    # Remove AI name prefix if present
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        logger.warning(f"TTS engine error: {e}")

# Create the Gradio interface
def create_interface():
    # Load configuration
    config = load_config()
    soulprint = load_soulprint()
    
    # Initialize services
    initialize_services()
    
    # Define theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    ) if config["theme"] == "light" else gr.themes.Soft(
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
    
    # Get available models
    available_models = model_service.get_available_models() if model_service else []
    available_providers = list(model_service.providers.keys()) if model_service else ["Ollama", "vLLM", "API"]
    
    with gr.Blocks(theme=theme, title=f"{config['active_ai'].capitalize()} Platinum Mission Control") as interface:
        # Header
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Image("public/assets/viren_logo.png" if os.path.exists("public/assets/viren_logo.png") else None, 
                         show_label=False, container=False, height=100)
            
            with gr.Column(scale=3):
                gr.Markdown(f"# {config['active_ai'].capitalize()} Platinum Mission Control")
                gr.Markdown(f"*{soulprint.get('purpose', 'Advanced AI System Management Interface')}*")
            
            with gr.Column(scale=1):
                status_indicator = gr.Textbox(
                    value=f"{config['active_ai'].capitalize()} is online - {config['deployment']} mode", 
                    label="Status", 
                    interactive=False
                )
        
        # Main tabs
        with gr.Tabs() as tabs:
            # Chat Tab
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(height=500, show_label=False)
                
                with gr.Row():
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            container=False
                        )
                    
                    with gr.Column(scale=1):
                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary")
                            voice_btn = gr.Button("🎤 Voice")
                            clear_btn = gr.Button("Clear")
            
            # Voice Tab
            with gr.TabItem("🎤 Voice"):
                with gr.Row():
                    with gr.Column(scale=2):
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
                        
                        voice_response = gr.Textbox(
                            label="AI Response",
                            interactive=False
                        )
                    
                    with gr.Column(scale=3):
                        video_output = gr.Video(
                            ANIMATION_PATH if os.path.exists(ANIMATION_PATH) else None,
                            label="AI Response Animation",
                            visible=config["animation_enabled"],
                            height=400
                        )
            
            # Dashboard Tab
            with gr.TabItem("📊 Dashboard"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### System Diagnostics")
                        
                        with gr.Row():
                            with gr.Column():
                                cpu_gauge = gr.Number(value=0, label="CPU Usage (%)")
                            with gr.Column():
                                memory_gauge = gr.Number(value=0, label="Memory Usage (%)")
                            with gr.Column():
                                disk_gauge = gr.Number(value=0, label="Disk Usage (%)")
                        
                        system_info = gr.JSON(
                            value={
                                "System": os.name,
                                "Uptime": "0:00:00",
                                "Active Model": config["active_model"],
                                "Provider": config["model_provider"]
                            },
                            label="System Information"
                        )
                        
                        refresh_btn = gr.Button("Refresh Dashboard")
                    
                    with gr.Column(scale=2):
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
                    
                    model_info = gr.JSON(
                        value={},
                        label="Active Model Information"
                    )
            
            # Gray's Anatomy Tab
            with gr.TabItem("🧠 Gray's Anatomy"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### System Visualization")
                        system_graph = gr.Plot(label="Component Relationships")
                        
                        update_graph_btn = gr.Button("Update Visualization")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Component Health History")
                        health_history_plot = gr.Plot(label="Health History")
                        
                        component_selector = gr.Dropdown(
                            choices=["All Components"],
                            value="All Components",
                            label="Select Component"
                        )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Settings")
                        
                        provider_dropdown = gr.Dropdown(
                            available_providers,
                            label="Model Provider",
                            value=config["model_provider"]
                        )
                        
                        model_dropdown = gr.Dropdown(
                            available_models,
                            label="Active Model",
                            value=config["active_model"]
                      
                        
                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=config["temperature"],
                                label="Temperature"
                            )
                            
                            max_tokens_slider = gr.Slider(
                                minimum=256,
                                maximum=4096,
                                step=256,
                                value=config["max_tokens"],
                                label="Max Tokens"
                            )
                    
                    with gr.Column():
                        gr.Markdown("### Interface Settings")
                        
                        theme_radio = gr.Radio(
                            ["light", "dark"],
                            label="Theme",
                            value=config["theme"]
                        )
                        
                        with gr.Row():
                            voice_checkbox = gr.Checkbox(
                                value=config["voice_enabled"],
                                label="Enable Voice"
                            )
                            
                            animation_checkbox = gr.Checkbox(
                                value=config["animation_enabled"],
                                label="Enable Animation"
                            )
                
                with gr.Row():
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                    reset_settings_btn = gr.Button("Reset to Defaults")
        
        # Event handlers
        def chat_response(message, history):
            if not message.strip():
                return history
            
            response = process_message(message, history)
            threading.Thread(target=speak_text, args=(response,)).start()
            
            # Return a list of tuples for Gradio chatbot component
            return history + [[message, response]]
        
        submit_btn.click(
            fn=chat_response,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            queue=False
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            fn=chat_response,
            inputs=[msg, chatbot],
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
            return text, response
        
        audio_input.change(
            fn=handle_voice_input,
            inputs=[audio_input],
            outputs=[voice_output, voice_response]
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
                "Active Model": config["active_model"],
                "Provider": config["model_provider"],
                "Deployment": config["deployment"],
                "Python Version": sys.version.split()[0]
            }
            
            # Update logs
            logs = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dashboard refreshed\n"
            logs += f"CPU: {metrics['cpu']}%, Memory: {metrics['memory']}%, Disk: {metrics['disk']}%\n"
            logs += f"Network: Sent {metrics['network']['bytes_sent']/1024:.2f} KB, Received {metrics['network']['bytes_recv']/1024:.2f} KB\n"
            
            # Get model info
            model_info_data = model_service.get_model_info(config["active_model"]) if model_service else {}
            
            return metrics["cpu"], metrics["memory"], metrics["disk"], system_info_data, fig, logs, model_info_data
        
        refresh_btn.click(
            fn=update_dashboard,
            outputs=[cpu_gauge, memory_gauge, disk_gauge, system_info, performance_plot, system_logs, model_info]
        )
        
        def update_grays_anatomy():
            if not anatomy_service:
                return [], None, {}
            
            # Get components
            components = anatomy_service.components
            
            # Create dataframe data
            df_data = []
            for name, data in components.items():
                connections = ", ".join(data["connections"])
                df_data.append([name, data["health"], data["type"], connections])
            
            # Create graph
            fig = anatomy_service.create_system_graph()
            
            # Component details (first component as default)
            first_component = next(iter(components.items()))
            details = {
                "name": first_component[0],
                **first_component[1]
            }
            
            # Update component selector
            component_names = ["All Components"] + list(components.keys())
            
            # Create health history graph
            history_fig = anatomy_service.create_health_history_graph()
            
            return df_data, fig, details, component_names, history_fig
        
        update_graph_btn.click(
            fn=update_grays_anatomy,
            outputs=[component_list, system_graph, component_details, component_selector, health_history_plot]
        )
        
        def update_component_history(component_name):
            if not anatomy_service:
                return None
            
            if component_name == "All Components":
                fig = anatomy_service.create_health_history_graph()
            else:
                fig = anatomy_service.create_health_history_graph(component_name)
            
            return fig
        
        component_selector.change(
            fn=update_component_history,
            inputs=[component_selector],
            outputs=[health_history_plot]
        )
        
        def update_model_options(provider):
            if not model_service:
                return gr.Dropdown.update(choices=[], value=None)
            
            models = model_service.get_available_models(provider=provider)
            return gr.Dropdown.update(choices=models, value=models[0] if models else None)
        
        provider_dropdown.change(
            fn=update_model_options,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        
        def save_settings(provider, model, temperature, max_tokens, theme, voice_enabled, animation_enabled):
            config = load_config()
            config["model_provider"] = provider
            config["active_model"] = model
            config["temperature"] = temperature
            config["max_tokens"] = max_tokens
            config["theme"] = theme
            config["voice_enabled"] = voice_enabled
            config["animation_enabled"] = animation_enabled
            
            save_config(config)
            
            # Set active model
            if model_service:
                model_service.set_active_model(model, provider)
            
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
            outputs=[gr.Textbox(visible=True)]
        )
        
        def reset_settings():
            save_config(DEFAULT_CONFIG)
            
            return (
                DEFAULT_CONFIG["model_provider"],
                DEFAULT_CONFIG["active_model"],
                DEFAULT_CONFIG["temperature"],
                DEFAULT_CONFIG["max_tokens"],
                DEFAULT_CONFIG["theme"],
                DEFAULT_CONFIG["voice_enabled"],
                DEFAULT_CONFIG["animation_enabled"],
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
                gr.Textbox(visible=True)
            ]
        )
        
        # Initialize Gray's Anatomy on load
        interface.load(
            fn=update_grays_anatomy,
            outputs=[component_list, system_graph, component_details, component_selector, health_history_plot]
        )
        
        # Initialize dashboard on load
        interface.load(
            fn=update_dashboard,
            outputs=[cpu_gauge, memory_gauge, disk_gauge, system_info, performance_plot, system_logs, model_info]
        )
    
    return interface

# Main function
def main():
    logger.info("Starting Viren Platinum Mission Control Panel")
    
    # Create interface
    interface = create_interface()
    
    # Launch interface
    interface.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True,
        favicon_path="public/64xAetherealCube.ico" if os.path.exists("public/64xAetherealCube.ico") else None
    )

if __name__ == "__main__":
    main()
