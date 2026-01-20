import gradio as gr
import os

# Company colors
LIGHT_BLUE = "#7EB6FF"
LIGHT_PURPLE = "#C8A2C8"
SILVER = "#E8E8E8"
WHITE = "#FFFFFF"

# Path to animation
ANIMATION_PATH = "public/morph_orb.mp4"

# Create public directory if it doesn't exist
os.makedirs("public", exist_ok=True)

# Simple chat function
def chat(message, history):
    return f"You said: {message}"

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
    """
) as demo:
    # Main interface
    with gr.Tab("Chat"):
        gr.Markdown("# Viren AI Assistant")
        
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
        
        # Handle text input
        msg.submit(chat, [msg, chatbot], [msg, chatbot])
        submit_btn.click(chat, [msg, chatbot], [msg, chatbot])
        
        # Switch to voice tab when mic button is clicked
        mic_btn.click(lambda: gr.Tabs.update(selected=1), None, None)
    
    # Voice interface
    with gr.Tab("Voice"):
        gr.Markdown("# Viren Voice Assistant")
        
        with gr.Row():
            with gr.Column():
                # Video display
                video = gr.Video(
                    ANIMATION_PATH if os.path.exists(ANIMATION_PATH) else None,
                    label="Viren",
                    height=300,
                    width=300
                )
            
            with gr.Column():
                # Audio input
                audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speak to Viren"
                )
                
                # Response display
                response = gr.Textbox(label="Viren's Response")
                
                # Back button
                back_btn = gr.Button("Back to Chat")
        
        # Process audio input
        def process_audio(audio_path):
            if not audio_path:
                return "Please speak into the microphone", None
            
            # In a real implementation, this would transcribe the audio
            # and generate a response
            return "I heard you speaking!", ANIMATION_PATH
        
        audio.change(process_audio, [audio], [response, video])
        
        # Switch back to chat tab
        back_btn.click(lambda: gr.Tabs.update(selected=0), None, None)

# Launch the interface
if __name__ == "__main__":
    demo.launch()
