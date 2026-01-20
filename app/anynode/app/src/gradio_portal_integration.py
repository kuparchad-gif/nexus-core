#!/usr/bin/env python3
"""
Gradio Portal Integration - Serves HTML portals within Gradio interface
Uses Gradio's port (7860) instead of conflicting with existing infrastructure
"""

import gradio as gr
import os
from enhanced_viren_api import process_viren_message

def load_html_file(filename):
    """Load HTML file content"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f'<h1>File not found: {filename}</h1>'

def create_portal_interface():
    """Create Gradio interface with HTML portals"""
    
    with gr.Blocks(title="Viren Portal System") as interface:
        gr.Markdown("# 🤖 Viren Portal System")
        
        with gr.Tabs():
            with gr.TabItem("Main Portal"):
                portal_html = gr.HTML(load_html_file('viren_portal.html'))
            
            with gr.TabItem("Dashboard"):
                dashboard_html = gr.HTML(load_html_file('viren_dashboard.html'))
            
            with gr.TabItem("Orb Interface"):
                orb_html = gr.HTML(load_html_file('viren_orb_dashboard.html'))
            
            with gr.TabItem("Orb Ultimate"):
                orb_ultimate_html = gr.HTML(load_html_file('viren_orb_ultimate.html'))
            
            with gr.TabItem("Network Status"):
                with gr.Row():
                    with gr.Column():
                        status_display = gr.JSON(label="Horn Network Status")
                        refresh_btn = gr.Button("Refresh Status")
                        
                def get_network_status():
                    try:
                        import requests
                        response = requests.get("http://viren-master:333/network_status", timeout=5)
                        return response.json() if response.status_code == 200 else {"error": "Master horn offline"}
                    except:
                        return {"error": "Cannot connect to master horn"}
                
                refresh_btn.click(get_network_status, outputs=status_display)
                interface.load(get_network_status, outputs=status_display)
            
            with gr.TabItem("Chat Interface"):
                chatbot = gr.Chatbot()
                msg = gr.Textbox(placeholder="Type your message...")
                
                def respond(message, history):
                    response = process_viren_message(message)
                    history.append((message, response))
                    return history, ""
                
                msg.submit(respond, [msg, chatbot], [chatbot, msg])
    
    return interface

if __name__ == "__main__":
    interface = create_portal_interface()
    interface.launch(server_port=7860, share=False)