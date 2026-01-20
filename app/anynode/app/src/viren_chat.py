# viren_chat.py
import gradio as gr
import os
import time
import subprocess
import threading

# Company colors
LIGHT_BLUE = "#7EB6FF"
LIGHT_PURPLE = "#C8A2C8"  # Plum-like color
SILVER = "#E8E8E8"
WHITE = "#FFFFFF"

# Ensure the animation file exists
ANIMATION_PATH = "public/morph_orb.mp4"

def chat_with_viren(message, history):
    """Chat with Viren."""
    if not message:
        return "", history
    
    # Simple response for demonstration
    response = f"I understand your question about: {message}"
    
    return "", history + [[message, response]]

def transcribe_audio(audio_path):
    """Transcribe audio to text."""
    if not audio_path:
        return "No audio detected"
    
    # In a real implementation, this would use speech recognition
    # For now, just return a placeholder
    return "Audio detected. This would be transcribed text."

def text_to_speech(text):
    """Convert text to speech."""
    # In a real implementation, this would generate audio
    # For now, just return success
    time.sleep(1)  # Simulate processing time
    return True

def process_voice(audio_path, history):
    """Process voice input and update history."""
    if not audio_path:
        return history, None
    
    # Transcribe audio
    transcribed_text = transcribe_audio(audio_path)
    
    # Get response from Viren
    response = f"I heard: {transcribed_text}"
    
    # Update history
    new_history = history + [[transcribed_text, response]]
    
    # Convert response to speech (in a real implementation)
    text_to_speech(response)
    
    return new_history, ANIMATION_PATH

def create_interface():
    """Create the Gradio interface."""
    # Custom CSS for company colors
    custom_css = f"""
    .gradio-container {{
        background: linear-gradient(135deg, {WHITE} 0%, {SILVER} 100%);
    }}

    .chat-message.user {{
        background-color: {LIGHT_BLUE};
        color: white;
    }}

    .chat-message.bot {{
        background-color: {LIGHT_PURPLE};
        color: white;
    }}

    h1, h3 {{
        color: {LIGHT_PURPLE} !important;
    }}

    .mic-button {{
        font-size: 24px !important;
        padding: 10px !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: {LIGHT_PURPLE} !important;
        color: white !important;
    }}
    """
    
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# Viren AI Assistant")
        
        # Shared state
        chatbot_state = gr.State([])
        
        # Main chat interface
        with gr.Row(visible=True) as chat_interface:
            with gr.Column():
                chatbot = gr.Chatbot(value=[], height=400)
                
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your message here...", scale=8)
                    send_btn = gr.Button("Send", scale=1)
                    mic_btn = gr.Button("🎤", elem_classes=["mic-button"], scale=1)
        
        # Voice interface (initially hidden)
        with gr.Row(visible=False) as voice_interface:
            with gr.Column(scale=2):
                video = gr.Video(ANIMATION_PATH, autoplay=False, width=400, height=400)
            
            with gr.Column(scale=3):
                voice_chatbot = gr.Chatbot(value=[], height=300)
                audio_input = gr.Audio(sources=["microphone"], type="filepath")
                back_btn = gr.Button("Back to Chat")
        
        # Handle text chat
        send_btn.click(
            chat_with_viren,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        msg.submit(
            chat_with_viren,
            [msg, chatbot],
            [msg, chatbot]
        )
        
        # Handle microphone button - show voice interface, hide chat
        def switch_to_voice():
            return gr.update(visible=False), gr.update(visible=True)
        
        mic_btn.click(
            switch_to_voice,
            [],
            [chat_interface, voice_interface]
        )
        
        # Handle back button - show chat, hide voice interface
        def switch_to_chat(history):
            return gr.update(visible=True), gr.update(visible=False), history
        
        back_btn.click(
            switch_to_chat,
            [voice_chatbot],
            [chat_interface, voice_interface, chatbot]
        )
        
        # Handle audio input
        audio_input.change(
            process_voice,
            [audio_input, voice_chatbot],
            [voice_chatbot, video]
        )
    
    return demo

if __name__ == "__main__":
    # Check if animation file exists
    if not os.path.exists(ANIMATION_PATH):
        print(f"Warning: Animation file not found at {ANIMATION_PATH}")
        print("Creating a placeholder file...")
        
        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(ANIMATION_PATH), exist_ok=True)
        
        # Create a placeholder file or copy from another location if available
        try:
            # Try to find an existing video file
            video_files = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith((".mp4", ".avi", ".mov")):
                        video_files.append(os.path.join(root, file))
            
            if video_files:
                # Copy the first video file found
                import shutil
                shutil.copy2(video_files[0], ANIMATION_PATH)
                print(f"Copied {video_files[0]} to {ANIMATION_PATH}")
            else:
                # Create an empty file as placeholder
                with open(ANIMATION_PATH, "wb") as f:
                    f.write(b"")
                print(f"Created empty placeholder at {ANIMATION_PATH}")
        except Exception as e:
            print(f"Error creating placeholder: {e}")
    
    # Launch the interface
    demo = create_interface()
    demo.launch()
