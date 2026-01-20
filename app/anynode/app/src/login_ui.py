#!/usr/bin/env python3
"""
Login UI for Cloud Viren
Provides a secure login interface
"""

import gradio as gr
from auth import Auth

def create_login_ui(on_login_success):
    """Create a login UI that matches the Cloud Viren aesthetic"""
    
    auth = Auth()
    
    with gr.Blocks(css=get_login_css()) as login_ui:
        # Orb video background
        gr.HTML("""
        <div id='orb-background'>
            <video id='orb-video' autoplay loop muted playsinline style="position: fixed; width: 100vw; height: 100vh; object-fit: cover; z-index: -2;">
                <source src="https://storage.googleapis.com/viren-assets/morph_orb.mp4" type="video/mp4">
            </video>
        </div>
        """)
        
        with gr.Column(elem_id="login-container"):
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
                
                error_message = gr.Markdown(elem_id="error-message")
            
            def login_attempt(username, password):
                success, token, message = auth.login(username, password)
                if success:
                    return {"visible": False, "__token__": token}
                else:
                    return {"visible": True, "value": f"**Error:** {message}"}
            
            login_button.click(
                fn=login_attempt,
                inputs=[username, password],
                outputs=[error_message],
                _js="(result) => { if(result.__token__) { localStorage.setItem('viren_token', result.__token__); window.location.reload(); } return result; }"
            )
    
    return login_ui

def get_login_css():
    """Get CSS for the login UI"""
    return """
    /* Base background container */
    #login-container {
        height: 100vh;
        width: 100vw;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    #login-title {
        color: white;
        text-align: center;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        margin-bottom: 20px;
    }
    
    /* Transparent glass-like login form */
    #login-form {
        width: 400px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        padding: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
        z-index: 2;
    }
    
    #username-input, #password-input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
        margin-bottom: 15px;
    }
    
    #login-button {
        background: linear-gradient(135deg, #A2799A, #93AEC5);
        border-radius: 20px;
        border: none;
        padding: 10px;
        color: white;
        font-weight: bold;
        margin-top: 10px;
        box-shadow: 0 0 15px rgba(162, 121, 154, 0.5);
    }
    
    #error-message {
        color: #ff6b6b;
        margin-top: 15px;
        text-align: center;
    }
    
    /* Orb video */
    #orb-video {
        opacity: 0.15;
        transition: opacity 0.5s ease;
    }
    """