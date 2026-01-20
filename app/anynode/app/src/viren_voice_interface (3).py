# viren_voice_interface.py
import gradio as gr
import os
import time
import random

# Company colors
LIGHT_BLUE = "#7EB6FF"
LIGHT_PURPLE = "#C8A2C8"  # Plum-like color
SILVER = "#E8E8E8"
WHITE = "#FFFFFF"

# Path to animation
ANIMATION_PATH = "public/morph_orb.mp4"

def chat_with_viren(message, history):
    """Chat with Viren."""
    if not message:
        return "", history
    
    # Simple response for demonstration
    response = f"I understand your question about: {message}"
    
    return "", history + [[message, response]]

def process_audio(audio_path):
    """Process audio input and return response with animation."""
    if not audio_path:
        return "No audio detected", None
    
    # In a real implementation, this would transcribe the audio
    # For now, just return a placeholder response
    response = "I heard you speaking! This is my response to what you said."
    
    # Return the response and the animation path
    # The animation will play while Viren "speaks"
    return response, ANIMATION_PATH

# Create a simple interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    css=f"""
    .gradio-container {{
        background: linear-gradient(135deg, {WHITE} 0%, {SILVER} 100%);
    }}
    .mic-button {{
        font-size: 24px !important;
        background-color: {LIGHT_PURPLE} !important;
        color: white !important;
        border-radius: 50% !important;
    }}
    .video-overlay {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1000;
        background: rgba(0,0,0,0.7);
        padding: 20px;
        border-radius: 10px;
        display: none;
    }}
    .video-overlay.active {{
        display: block;
    }}
    """
) as demo:
    gr.Markdown("# Viren AI Assistant")
    
    # Create a hidden video overlay div that will be shown when Viren speaks
    with gr.Row(elem_classes=["video-overlay"], visible=False) as video_overlay:
        video = gr.Video(
            ANIMATION_PATH,
            autoplay=True,
            loop=True,
            width=300,
            height=300
        )
    
    # Main chat interface
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=400, type="messages")
            
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Send")
                with gr.Column(scale=1):
                    mic_btn = gr.Button("🎤", elem_classes=["mic-button"])
    
    # Audio input (initially hidden)
    with gr.Row(visible=False) as audio_row:
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak to Viren"
        )
        response_text = gr.Textbox(
            label="Viren's Response",
            interactive=False
        )
    
    # Handle text input
    msg.submit(chat_with_viren, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat_with_viren, [msg, chatbot], [msg, chatbot])
    
    # Handle microphone button - show audio input
    def toggle_audio():
        return gr.update(visible=True), gr.update(visible=False)
    
    mic_btn.click(toggle_audio, [], [audio_row, mic_btn])
    
    # Handle audio input - process and show animation overlay
    def show_animation(response, animation_path):
        # Show the video overlay
        return gr.update(visible=True), response
    
    audio_input.change(
        process_audio,
        [audio_input],
        [response_text, video]
    ).then(
        show_animation,
        [response_text, video],
        [video_overlay, response_text]
    )
    
    # Hide animation after response
    def hide_animation():
        time.sleep(5)  # Keep animation visible for 5 seconds
        return gr.update(visible=False), gr.update(visible=True)
    
    response_text.change(
        hide_animation,
        [],
        [video_overlay, mic_btn]
    )

# Launch the interface
if __name__ == "__main__":
    # Create public directory if it doesn't exist
    os.makedirs("public", exist_ok=True)
    
    # Check if animation file exists
    if not os.path.exists(ANIMATION_PATH):
        print(f"Warning: Animation file not found at {ANIMATION_PATH}")
        print("Please ensure the file exists before running this interface.")
    
    demo.launch()
