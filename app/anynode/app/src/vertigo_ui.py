import gradio as gr
from viren_mind import route_and_merge

def command_bridge(prompt):
    logs, result = route_and_merge(prompt)
    return logs, result

with gr.Blocks() as demo:
    gr.Markdown("# 🧬 VIREN MCP Interface | VERTIGO Online")
    prompt = gr.Textbox(label="Prompt")
    route_log = gr.Textbox(label="Routing Log", lines=3)
    output = gr.Textbox(label="Model Output", lines=4)
    prompt.submit(fn=command_bridge, inputs=prompt, outputs=[route_log, output])

demo.launch(server_name="0.0.0.0", server_port=7860)
