import gradio as gr
import time
import sys
import os
from pathlib import Path

# Add Systems to path for bridge components
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Import bridge components
try:
    from Systems.address_manager.fluxtether.flux_core import FluxCore
    from Systems.address_manager.pulse13 import Pulse13
    bridge_available = True
except ImportError as e:
    print(f"Bridge components not available: {e}")
    bridge_available = False

# Initialize bridge if available
if bridge_available:
    flux_core = FluxCore()
    pulse13 = Pulse13()
else:
    flux_core = None
    pulse13 = None

def user(user_msg, history):
    return "", history + [[user_msg, None]]

def bot(history):
    user_message = history[-1][0] if history else ""
    
    # Use bridge components if available
    if bridge_available and flux_core:
        try:
            response = flux_core.process_message(user_message)
        except:
            response = "Viren is online. Full services running. How can I help?"
    else:
        response = "Viren is online. Full services running. How can I help?"
    
    if history and history[-1][1] is None:
        history[-1][1] = ""
        for char in response:
            history[-1][1] += char
            time.sleep(0.02)
            yield history

# Custom CSS with orb background and theme
custom_css = """
body, #root {
    background: linear-gradient(135deg, #0a0a0a, #1a1a2e, #16213e) !important;
    color: white !important;
}

#orb-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -2;
    overflow: hidden;
}

#orb-video {
    position: absolute;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    width: auto;
    height: auto;
    transform: translate(-50%, -50%);
    object-fit: cover;
}

#background-container {
    background: rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    margin: 20px;
}

#chat-area {
    background: rgba(0, 0, 0, 0.5) !important;
    border-radius: 15px;
    padding: 20px;
}

#chatbot {
    background: rgba(0, 0, 0, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

.message {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px;
    margin: 5px 0;
    padding: 10px;
}

#voice-buttons {
    position: fixed;
    bottom: 20px;
    right: 20px;
    display: flex;
    gap: 10px;
    z-index: 1000;
}

.circle-button {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: rgba(74, 105, 189, 0.8);
    color: white;
    border: none;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.3s;
}

.circle-button:hover {
    background: rgba(74, 105, 189, 1);
    transform: scale(1.1);
}
"""

with gr.Blocks(css=custom_css, title="Viren AI") as demo:
    # Orb video background
    gr.HTML("""
    <div id='orb-background'>
        <video id='orb-video' autoplay loop muted playsinline>
            <source src="morph_orb.mp4" type="video/mp4">
            <div style="background: radial-gradient(circle, #4a69bd, #1a1a2e); width: 100%; height: 100%;"></div>
        </video>
    </div>
    """)

    with gr.Column(elem_id="background-container"):
        with gr.Column(elem_id="chat-area"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=400
            )

            chat_input = gr.Textbox(
                placeholder="Enter message...",
                show_label=False,
                elem_id="chat-input"
            )

            chat_input.submit(user, [chat_input, chatbot], [chat_input, chatbot], queue=False)\
                      .then(bot, chatbot, chatbot)

    # Voice control buttons
    gr.HTML("""
    <div id="voice-buttons">
        <button id="mic-toggle" class="circle-button">MIC</button>
        <button id="upload-btn" class="circle-button">UPLOAD</button>
        <button id="exit-btn" class="circle-button">EXIT</button>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)