#!/usr/bin/env python3
"""
Cloud Viren Deployment Script
Downloads models and sets up the environment on Modal
"""

import os
import sys
import modal
from pathlib import Path

# Create Modal app
app = modal.App("viren-cloud")

# Create persistent volumes
models_volume = modal.Volume.from_name("viren-models-volume", create_if_missing=True)
binary_volume = modal.Volume.from_name("binary-protocol-volume", create_if_missing=True)
weaviate_volume = modal.Volume.from_name("weaviate-data-volume", create_if_missing=True)
assets_volume = modal.Volume.from_name("viren-assets-volume", create_if_missing=True)

# Base image with dependencies
image = modal.Image.debian_slim().pip_install(
    "weaviate-client>=3.15.0",
    "sentence-transformers>=2.2.2",
    "transformers>=4.26.0",
    "torch>=1.13.0",
    "fastapi>=0.95.0",
    "uvicorn>=0.21.0",
    "python-dotenv>=1.0.0",
    "pydantic>=1.10.7",
    "gradio>=3.50.2",
    "websockets>=11.0.3",
    "huggingface_hub>=0.19.0",
    "safetensors>=0.4.0"
)

# Define model setup function
@app.function(
    image=image,
    volumes={"/app/models": models_volume},
    timeout=1800,  # 30 minutes timeout for copying models
    gpu="A10G"
)
def setup_models():
    """Copy local models to the models volume"""
    import os
    import shutil
    import glob
    
    # Create models directory
    os.makedirs("/app/models", exist_ok=True)
    
    results = {}
    
    # Check if running locally
    if os.path.exists("C:/Viren/models"):
        # Get list of model directories
        model_dirs = [d for d in os.listdir("C:/Viren/models") if os.path.isdir(os.path.join("C:/Viren/models", d))]
        
        # Copy each model directory
        for model_dir in model_dirs:
            src_dir = os.path.join("C:/Viren/models", model_dir)
            dst_dir = os.path.join("/app/models", model_dir)
            
            # Skip if already exists
            if os.path.exists(dst_dir):
                results[model_dir] = "Already exists, skipped copy"
                continue
            
            try:
                print(f"Copying model {model_dir}...")
                shutil.copytree(src_dir, dst_dir)
                results[model_dir] = "Copied successfully"
            except Exception as e:
                results[model_dir] = f"Error copying: {str(e)}"
        
        # Copy model manifest if it exists
        manifest_path = os.path.join("C:/Viren/models", "model_manifest.json")
        if os.path.exists(manifest_path):
            try:
                shutil.copy2(manifest_path, os.path.join("/app/models", "model_manifest.json"))
                results["model_manifest.json"] = "Copied successfully"
            except Exception as e:
                results["model_manifest.json"] = f"Error copying: {str(e)}"
    else:
        # If not running locally, create placeholder models
        os.makedirs("/app/models/placeholder", exist_ok=True)
        with open("/app/models/placeholder/info.txt", "w") as f:
            f.write("Placeholder model - real models need to be uploaded")
        results["placeholder"] = "Created placeholder model"
    
    return results

# Define assets setup function
@app.function(
    image=image,
    volumes={"/app/assets": assets_volume},
    timeout=300
)
def setup_assets():
    """Set up assets for Cloud Viren UI"""
    import os
    import shutil
    
    # Create assets directory
    os.makedirs("/app/assets", exist_ok=True)
    
    # Copy assets from local directory if running locally
    if os.path.exists("C:/Viren/cloud/assets"):
        for item in os.listdir("C:/Viren/cloud/assets"):
            src_path = os.path.join("C:/Viren/cloud/assets", item)
            dst_path = os.path.join("/app/assets", item)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied asset: {item}")
    
    # Create placeholder files if needed
    if not os.path.exists("/app/assets/morph_orb.mp4"):
        with open("/app/assets/morph_orb.mp4", "wb") as f:
            f.write(b"PLACEHOLDER_VIDEO")
        print("Created placeholder video")
    
    if not os.path.exists("/app/assets/style.css"):
        with open("/app/assets/style.css", "w") as f:
            f.write("""
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #000;
            }
            
            #background-container {
                height: 100vh;
                width: 100vw;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            
            #chat-area {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 30px;
                padding: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
                width: 80%;
                max-width: 1000px;
            }
            
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
            """)
        print("Created style.css")
    
    return {"status": "Assets setup complete"}

# Define Weaviate container
@app.container(
    image="semitechnologies/weaviate:1.24.5",
    ports=[8080],
    volumes={"/var/lib/weaviate": weaviate_volume},
    env={
        "QUERY_DEFAULTS_LIMIT": "25",
        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
        "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
        "DEFAULT_VECTORIZER_MODULE": "text2vec-transformers",
        "ENABLE_MODULES": "text2vec-transformers",
        "TRANSFORMERS_INFERENCE_API": "http://t2v-transformers:8080",
        "CLUSTER_HOSTNAME": "node1"
    }
)
def weaviate_container():
    pass

# Define transformers container
@app.container(
    image="semitechnologies/transformers-inference:sentence-transformers-multi-qa-mpnet-base-dot-v1",
    env={
        "ENABLE_CUDA": "0"
    }
)
def t2v_transformers():
    pass

# Define API endpoint
@app.function(
    image=image,
    volumes={
        "/app/models": models_volume,
        "/app/memory": binary_volume
    },
    timeout=300,
    secrets=[modal.Secret.from_name("viren-api-keys")],
    gpu="A10G"
)
@modal.asgi_app()
def viren_api():
    """API for Viren Cloud"""
    from fastapi import FastAPI, HTTPException, Depends, Header
    from pydantic import BaseModel
    import os
    
    app = FastAPI(title="Viren Cloud API")
    
    # Authentication
    def verify_api_key(x_api_key: str = Header(None)):
        valid_keys = os.environ.get("VIREN_API_KEYS", "").split(",")
        if not x_api_key or x_api_key not in valid_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key
    
    class QueryRequest(BaseModel):
        query: str
        model_id: str = None
        
    @app.get("/")
    def read_root():
        return {"status": "Viren Cloud API is running"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    @app.post("/query")
    def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
        try:
            # Process query using selected model
            # In a real implementation, this would route to the appropriate model
            return {
                "success": True, 
                "result": f"Processed query with model {request.model_id or 'default'}: {request.query}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    return app

# Define UI server
@app.function(
    image=image,
    volumes={
        "/app/models": models_volume,
        "/app/assets": assets_volume,
        "/app/plugins": modal.Volume.from_name("viren-plugins-volume", create_if_missing=True)
    },
    timeout=300,
    secrets=[modal.Secret.from_name("viren-api-keys")]
)
def cloud_ui_server():
    """Cloud Viren UI Server"""
    import gradio as gr
    import os
    import sys
    import json
    from gradio.themes.base import Base
    from gradio.themes.utils import colors, sizes
    
    # Add assets directory to path
    sys.path.append("/app/assets")
    
    # Import auth module
    class Auth:
        """Simple authentication handler"""
        
        def __init__(self):
            self.users = {
                "admin": {
                    "password": "admin123",
                    "role": "admin"
                }
            }
            self.sessions = {}
        
        def login(self, username, password):
            """Authenticate a user"""
            if username in self.users and self.users[username]["password"] == password:
                import uuid
                token = str(uuid.uuid4())
                self.sessions[token] = {
                    "username": username,
                    "role": self.users[username]["role"]
                }
                return True, token, "Login successful"
            return False, None, "Invalid username or password"
    
    # Create auth instance
    auth = Auth()
    
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
                    c500="#A2799A",
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
                    c500="#93AEC5",
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
    
    # Create custom CSS
    custom_css = """
    body {
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    
    #main-container {
        display: none;
        height: 100vh;
        width: 100vw;
        flex-direction: column;
        overflow: hidden;
    }
    
    #login-container {
        display: flex;
        height: 100vh;
        width: 100vw;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
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
    
    #orb-video {
        opacity: 0.15;
        transition: opacity 0.5s ease;
    }
    
    #orb-video.speaking {
        opacity: 0.6;
        filter: saturate(150%) brightness(1.3);
        transition: all 0.3s ease-in-out;
    }
    """
    
    # Create a simple UI
    with gr.Blocks(title="Cloud Viren", theme=GlassTheme(), css=custom_css) as demo:
        # Orb video background
        gr.HTML("""
        <div id='orb-background'>
            <video id='orb-video' autoplay loop muted playsinline style="position: fixed; width: 100vw; height: 100vh; object-fit: cover; z-index: -2;">
                <source src="/file=assets/morph_orb.mp4" type="video/mp4">
            </video>
        </div>
        """)
        
        # Login state
        is_logged_in = gr.State(False)
        username_state = gr.State("")
        
        # Login interface
        with gr.Row(elem_id="login-container") as login_row:
            with gr.Column():
                gr.Markdown("# Cloud Viren", elem_id="login-title")
                
                with gr.Column(elem_id="login-form"):
                    username = gr.Textbox(
                        label="Username",
                        placeholder="Enter username",
                        elem_id="username-input"
                    )
                    
                    password = gr.Textbox(
                        label="Password",
                        placeholder="Enter password",
                        type="password",
                        elem_id="password-input"
                    )
                    
                    login_button = gr.Button("Login", variant="primary", elem_id="login-button")
                    
                    login_message = gr.Markdown(elem_id="error-message")
        
        # Main interface (hidden initially)
        with gr.Row(visible=False, elem_id="main-container") as main_row:
            with gr.Column():
                with gr.Row(elem_id="header"):
                    gr.Markdown("# Cloud Viren", elem_id="title")
                    with gr.Column(elem_id="user-info"):
                        gr.HTML("<div>Logged in as: <span id='user-display'></span></div>")
                        logout_btn = gr.Button("Logout", elem_id="logout-btn")
                
                with gr.Tabs():
                    with gr.Tab("Chat", id="chat-tab"):
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
                                            choices=["LLaVA-34B", "Moshi Voice"],
                                            value="LLaVA-34B"
                                        )
                                        
                                        send_btn = gr.Button("Send", variant="primary")
                    
                    with gr.Tab("Models", id="models-tab"):
                        with gr.Column(elem_id="models-area"):
                            gr.Markdown("## AI Models")
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    models_json = gr.JSON(
                                        label="Available Models",
                                        value={
                                            "llava-v1.6-34b": {
                                                "name": "LLaVA v1.6 34B",
                                                "type": "vision-language",
                                                "loaded": True
                                            },
                                            "moshiko": {
                                                "name": "Moshi Voice Model",
                                                "type": "speech-to-speech",
                                                "loaded": True
                                            }
                                        }
                                    )
                    
                    with gr.Tab("Plugins", id="plugins-tab"):
                        with gr.Column(elem_id="plugins-area"):
                            gr.Markdown("## Available Plugins")
                            
                            plugins_json = gr.JSON(
                                label="Plugins",
                                value={
                                    "3D_Animation_Arena": {"loaded": True},
                                    "AI-Game-Creator": {"loaded": True},
                                    "anycoder": {"loaded": True},
                                    "background-removal": {"loaded": True},
                                    "chat-ui": {"loaded": True},
                                    "CLIP-Interrogator": {"loaded": True},
                                    "deepsite": {"loaded": True},
                                    "HierSpeech_TTS": {"loaded": True},
                                    "InstantMesh": {"loaded": True},
                                    "MusicGen": {"loaded": True},
                                    "stable-diffusion": {"loaded": True}
                                }
                            )
                    
                    with gr.Tab("System", id="system-tab"):
                        with gr.Column(elem_id="system-area"):
                            gr.Markdown("## System Status")
                            
                            system_status = gr.JSON(
                                label="Status",
                                value={
                                    "status": "running",
                                    "models_loaded": True,
                                    "api_available": True,
                                    "weaviate_status": "connected",
                                    "memory_usage": "2.1 GB / 16 GB",
                                    "gpu_usage": "4.3 GB / 24 GB"
                                }
                            )
        
        # Login handler
        def login(username, password):
            success, token, message = auth.login(username, password)
            if success:
                return {
                    login_row: gr.update(visible=False),
                    main_row: gr.update(visible=True),
                    username_state: username,
                    is_logged_in: True
                }
            else:
                return {
                    login_message: f"**Error:** {message}",
                    is_logged_in: False
                }
        
        login_button.click(
            login,
            inputs=[username, password],
            outputs=[login_row, main_row, username_state, is_logged_in, login_message]
        )
        
        # Logout handler
        def logout():
            return {
                login_row: gr.update(visible=True),
                main_row: gr.update(visible=False),
                username_state: "",
                is_logged_in: False,
                login_message: ""
            }
        
        logout_btn.click(
            logout,
            inputs=[],
            outputs=[login_row, main_row, username_state, is_logged_in, login_message]
        )
        
        # Chat handler
        def user_message(message, history):
            return "", history + [[message, None]]
        
        def bot_response(history, model_name):
            user_message = history[-1][0]
            response = f"Processing query with {model_name}: {user_message}"
            
            # Simulate typing effect
            history[-1][1] = ""
            for char in response:
                history[-1][1] += char
                yield history
        
        send_btn.click(user_message, [chat_input, chatbot], [chat_input, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown], chatbot
        )
        
        chat_input.submit(user_message, [chat_input, chatbot], [chat_input, chatbot], queue=False).then(
            bot_response, [chatbot, model_dropdown], chatbot
        )
    
    # Launch the interface
    demo.launch(server_name="0.0.0.0", server_port=7860)

# Define plugin setup function
@app.function(
    image=image,
    volumes={"/app/plugins": modal.Volume.from_name("viren-plugins-volume", create_if_missing=True)},
    timeout=600
)
def setup_plugins():
    """Set up plugins for Cloud Viren"""
    import os
    import shutil
    import sys
    
    # Create plugins directory
    os.makedirs("/app/plugins", exist_ok=True)
    
    # Copy plugins from local directory if running locally
    if os.path.exists("C:/Viren/cloud/Cloud UI Plugins"):
        # Get list of plugins
        plugins = [item for item in os.listdir("C:/Viren/cloud/Cloud UI Plugins") 
                  if os.path.isdir(os.path.join("C:/Viren/cloud/Cloud UI Plugins", item))]
        
        # Copy each plugin
        for plugin in plugins:
            src_dir = os.path.join("C:/Viren/cloud/Cloud UI Plugins", plugin)
            dst_dir = os.path.join("/app/plugins", plugin)
            
            # Skip if already exists
            if os.path.exists(dst_dir):
                continue
            
            # Copy plugin
            shutil.copytree(src_dir, dst_dir)
            print(f"Copied plugin: {plugin}")
        
        # Copy utility files
        utility_files = ['database.py', 'document_tools.py', 'memory.py', 
                       'module_Scanning.py', 'modules.py', 'system_scan.py', 'voice.py']
        
        for file in utility_files:
            src_file = os.path.join("C:/Viren/cloud/Cloud UI Plugins", file)
            dst_file = os.path.join("/app/plugins", file)
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied utility file: {file}")
    
    return {"status": "Plugins setup complete", "plugins_count": len(os.listdir("/app/plugins"))}

@app.local_entrypoint()
def main():
    print("Deploying Cloud Viren...")
    
    # Set up assets
    print("Setting up assets...")
    assets_result = setup_assets.remote()
    print("Assets setup result:", assets_result)
    
    # Set up plugins
    print("Setting up plugins...")
    plugins_result = setup_plugins.remote()
    print("Plugins setup result:", plugins_result)
    
    # Download models
    print("Downloading models...")
    model_results = download_models.remote()
    print("Model download results:", model_results)
    
    # Start containers
    print("Starting Weaviate and transformers containers...")
    weaviate_container.run()
    t2v_transformers.run()
    
    # Start UI server
    print("Starting Cloud Viren UI...")
    cloud_ui_server.run()
    
    print("Cloud Viren deployment complete!")
    print("API available at: https://viren-cloud--viren-api.modal.run")
    print("UI available at: https://viren-cloud--cloud-ui-server.modal.run")

if __name__ == "__main__":
    main()