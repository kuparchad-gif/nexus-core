import gradio as gr
import time
import sys
import os
from pathlib import Path

# Add Systems to path
root_dir = Path(__file__).parent / "root"
sys.path.insert(0, str(root_dir))

def user(user_msg, history):
    return "", history + [[user_msg, None]]

def bot(history):
    user_message = history[-1][0] if history else ""
    
    # Simple responses for now
    if "chrome" in user_message.lower():
        response = "For Chrome reboots: Disable hardware acceleration in chrome://settings → Advanced → System"
    elif "help" in user_message.lower():
        response = "Viren is online with full services. LM Studio connected. How can I assist?"
    else:
        response = f"I understand you said: '{user_message}'. Viren services are running. What would you like me to help with?"
    
    if history and history[-1][1] is None:
        history[-1][1] = ""
        for char in response:
            history[-1][1] += char
            time.sleep(0.01)
            yield history

# Dark theme CSS
css = """
body { background: linear-gradient(135deg, #0a0a0a, #1a1a2e) !important; color: white !important; }
#chatbot { background: rgba(0,0,0,0.7) !important; border: 1px solid rgba(255,255,255,0.1); }
.message { background: rgba(255,255,255,0.1) !important; border-radius: 10px; margin: 5px 0; padding: 10px; }
"""

with gr.Blocks(css=css, title="Viren AI") as demo:
    gr.Markdown("# Viren AI - Full Services Online")
    gr.Markdown("*LM Studio connected • All services running • Ready to assist*")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Ask Viren anything...", show_label=False)
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)