#!/usr/bin/env python3
"""
Viren Platinum Edition - Enhanced Interface
"""

import gradio as gr
import os
import json
import time
import logging
import sys
from datetime import datetime
import threading

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"viren_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VirenInterface")

# Import modules
try:
    from model_manager import ModelManager
    from document_processor import DocumentProcessor
    from auth_manager import AuthManager
    from viren_tts import VirenTTS
    from viren_stt import VirenSTT
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.info("Creating placeholder modules...")
    
    # Placeholder classes if imports fail
    class ModelManager:
        def __init__(self):
            self.active_model = "gemma:2b"
            self.active_provider = "Ollama"
        
        def get_available_models(self):
            return ["gemma:2b", "hermes:2-pro-llama-3-7b", "phi:3-mini-4k"]
        
        def process_message(self, message):
            return f"This is a placeholder response for: {message}"
    
    class DocumentProcessor:
        def process_document(self, file_path):
            return {"status": "placeholder", "content": "Document processing not implemented"}
    
    class AuthManager:
        def verify_user(self, username, password, otp=None):
            return True
    
    class VirenTTS:
        def speak(self, text):
            logger.info(f"TTS would say: {text}")
    
    class VirenSTT:
        def transcribe(self, audio_path):
            return "Speech transcription placeholder"

# Initialize services
model_manager = ModelManager()
doc_processor = DocumentProcessor()
auth_manager = AuthManager()
tts_engine = VirenTTS()
stt_engine = VirenSTT()

# Configuration
CONFIG_PATH = os.path.join("config", "viren_platinum_config.json")
DEFAULT_CONFIG = {
    "theme": "dark",
    "animation_enabled": True,
    "voice_enabled": True,
    "active_model": "gemma:2b",
    "active_provider": "Ollama",
    "max_chat_pages": 4,
    "auto_model_switching": True,
    "mfa_required": False
}

def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

# Custom theme for platinum look
def create_platinum_theme():
    """Create a premium platinum-compatible theme (Gradio 4+)"""
    return gr.themes.Soft(
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
        button_primary_text_color="white",
        slider_color="#3b82f6",
        slider_color_dark="#3b82f6"
    )

def create_login_interface():
    """Create the login interface"""
    with gr.Blocks(theme=create_platinum_theme(), title="Viren Platinum - Login") as login_interface:
        gr.Markdown("# Viren Platinum Edition")
        gr.Markdown("### Secure Authentication")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Logo placeholder
                gr.Image("public/assets/viren_logo.png" if os.path.exists("public/assets/viren_logo.png") else None, 
                         show_label=False, container=False)
            
            with gr.Column(scale=2):
                username = gr.Textbox(label="Username", placeholder="Enter your username")
                password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
                otp = gr.Textbox(label="One-Time Password", placeholder="Enter OTP if enabled", visible=load_config().get("mfa_required", False))
                
                with gr.Row():
                    login_button = gr.Button("Login", variant="primary")
                    demo_button = gr.Button("Demo Mode")
                
                login_message = gr.Markdown("")
        
        def login(username, password, otp=None):
            if auth_manager.verify_user(username, password, otp):
                return "Login successful! Redirecting to Viren Platinum..."
            else:
                return "Invalid credentials. Please try again."
        
        login_button.click(
            fn=login,
            inputs=[username, password, otp],
            outputs=[login_message]
        )
        
        demo_button.click(
            fn=lambda: "Entering demo mode...",
            outputs=[login_message]
        )
    
    return login_interface

def create_main_interface():
    """Create the main interface"""
    config = load_config()
    
    with gr.Blocks(theme=create_platinum_theme(), title="Viren Platinum") as main_interface:
        # Header
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image("public/assets/viren_logo.png" if os.path.exists("public/assets/viren_logo.png") else None, 
                         show_label=False, container=False, height=100)
            
            with gr.Column(scale=3):
                gr.Markdown("# Viren Platinum Mission Control")
                gr.Markdown("*Advanced AI System with Document Processing & Model Switching*")
            
            with gr.Column(scale=1):
                status = gr.Textbox(value="Online - Ready", label="Status", interactive=False)
        
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
                            voice_btn = gr.Button("🎤", elem_classes=["mic-button"])
                            clear_btn = gr.Button("Clear")
            
            # Voice Tab
            with gr.TabItem("🎤 Voice Assistant"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["microphone"], 
                            type="filepath",
                            label="Voice Input"
                        )
                        
                        voice_output = gr.Textbox(
                            label="Transcription", 
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        video_output = gr.Video(
                            "public/morph_orb.mp4" if os.path.exists("public/morph_orb.mp4") else None,
                            label="Viren Response",
                            interactive=False,
                            autoplay=True,
                            loop=True,
                            height=400
                        )
            
            # Document Tab
            with gr.TabItem("📄 Document Studio"):
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload Document",
                            file_types=[".pdf", ".docx", ".xlsx", ".txt", ".csv", ".vsdx"],
                            file_count="multiple"
                        )
                        
                        process_btn = gr.Button("Process Documents", variant="primary")
                    
                    with gr.Column():
                        doc_output = gr.JSON(
                            label="Document Analysis"
                        )
                
                with gr.Row():
                    doc_viewer = gr.HTML(
                        label="Document Viewer",
                        value="<div class='doc-viewer'>Upload a document to view its contents</div>"
                    )
            
            # Canvas Tab
            with gr.TabItem("🖌️ Design Canvas"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Design Tools")
                        
                        tool_selector = gr.Radio(
                            ["Select", "Draw", "Text", "Shape", "Connector"],
                            label="Tool",
                            value="Select"
                        )
                        
                        color_picker = gr.ColorPicker(
                            label="Color",
                            value="#3b82f6"
                        )
                    
                    with gr.Column(scale=3):
                        canvas_html = gr.HTML(
                            value="<div id='viren-canvas' class='design-canvas'>Canvas loading...</div>"
                        )
                
                with gr.Row():
                    save_design_btn = gr.Button("Save Design")
                    load_design_btn = gr.Button("Load Design")
                    export_btn = gr.Button("Export")
            
            # Model Lab Tab
            with gr.TabItem("🧠 Model Lab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        
                        provider_selector = gr.Dropdown(
                            ["Ollama", "vLLM", "LM Studio", "API"],
                            label="Provider",
                            value=config.get("active_provider", "Ollama")
                        )
                        
                        model_selector = gr.Dropdown(
                            model_manager.get_available_models(),
                            label="Model",
                            value=config.get("active_model", "gemma:2b")
                        )
                        
                        refresh_models_btn = gr.Button("Refresh Models")
                    
                    with gr.Column():
                        gr.Markdown("### Model Actions")
                        
                        download_model_btn = gr.Button("Download Model")
                        hot_swap_btn = gr.Button("Hot Swap Model", variant="primary")
                        model_status = gr.Textbox(value="Ready", label="Status", interactive=False)
                
                with gr.Row():
                    gr.Markdown("### Model Performance")
                    model_metrics = gr.DataFrame(
                        headers=["Model", "Provider", "Response Time", "Token Rate", "Memory Usage"],
                        value=[
                            ["gemma:2b", "Ollama", "0.5s", "45 t/s", "700MB"],
                            ["hermes:2-pro-llama-3-7b", "Ollama", "1.2s", "28 t/s", "4GB"]
                        ]
                    )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Interface Settings")
                        
                        theme_selector = gr.Radio(
                            ["light", "dark"],
                            label="Theme",
                            value=config.get("theme", "dark")
                        )
                        
                        animation_toggle = gr.Checkbox(
                            label="Enable Animations",
                            value=config.get("animation_enabled", True)
                        )
                        
                        voice_toggle = gr.Checkbox(
                            label="Enable Voice",
                            value=config.get("voice_enabled", True)
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Security Settings")
                        
                        mfa_toggle = gr.Checkbox(
                            label="Require MFA",
                            value=config.get("mfa_required", False)
                        )
                        
                        chat_pages = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=config.get("max_chat_pages", 4),
                            label="Max Chat Pages Before Logging"
                        )
                
                with gr.Row():
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                    reset_settings_btn = gr.Button("Reset to Defaults")
                    settings_status = gr.Textbox(value="", label="Status", interactive=False)
        
        # Event handlers
        def process_message(message, history):
            if not message.strip():
                return history
            
            response = model_manager.process_message(message)
            
            # Start TTS in a separate thread
            if config.get("voice_enabled", True):
                threading.Thread(target=tts_engine.speak, args=(response,)).start()
            
            # Return updated history
            return history + [[message, response]]
        
        submit_btn.click(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            fn=lambda: None,
            outputs=[chatbot]
        )
        
        def handle_voice_input(audio_path):
            if not audio_path:
                return "No audio detected", ""
            
            # Transcribe audio
            text = stt_engine.transcribe(audio_path)
            
            # Process message
            response = model_manager.process_message(text)
            
            # Start TTS in a separate thread
            if config.get("voice_enabled", True):
                threading.Thread(target=tts_engine.speak, args=(response,)).start()
            
            return text, response
        
        audio_input.change(
            fn=handle_voice_input,
            inputs=[audio_input],
            outputs=[voice_output, gr.Textbox(visible=False)]
        )
        
        def process_documents(files):
            if not files:
                return {"status": "error", "message": "No files uploaded"}
            
            results = []
            for file in files:
                result = doc_processor.process_document(file.name)
                results.append(result)
            
            return {"status": "success", "results": results}
        
        process_btn.click(
            fn=process_documents,
            inputs=[file_upload],
            outputs=[doc_output]
        )
        
        def save_settings(theme, animation_enabled, voice_enabled, mfa_required, max_chat_pages):
            config = load_config()
            config["theme"] = theme
            config["animation_enabled"] = animation_enabled
            config["voice_enabled"] = voice_enabled
            config["mfa_required"] = mfa_required
            config["max_chat_pages"] = int(max_chat_pages)
            
            save_config(config)
            return "Settings saved successfully"
        
        save_settings_btn.click(
            fn=save_settings,
            inputs=[theme_selector, animation_toggle, voice_toggle, mfa_toggle, chat_pages],
            outputs=[settings_status]
        )
        
        def reset_settings():
            save_config(DEFAULT_CONFIG)
            return (
                DEFAULT_CONFIG["theme"],
                DEFAULT_CONFIG["animation_enabled"],
                DEFAULT_CONFIG["voice_enabled"],
                DEFAULT_CONFIG["mfa_required"],
                DEFAULT_CONFIG["max_chat_pages"],
                "Settings reset to defaults"
            )
        
        reset_settings_btn.click(
            fn=reset_settings,
            outputs=[theme_selector, animation_toggle, voice_toggle, mfa_toggle, chat_pages, settings_status]
        )
    
    return main_interface

def main():
    """Main function to run the interface"""
    logger.info("Starting Viren Platinum Interface")
    
    config = load_config()
    
    # Check if MFA is required
    if config.get("mfa_required", False):
        login_interface = create_login_interface()
        login_interface.launch(prevent_thread_lock=True)
    
    # Create and launch main interface
    main_interface = create_main_interface()
    main_interface.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
