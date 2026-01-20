import gradio as gr
from modal_client import viren_router

def route_with_modal(prompt):
    return viren_router(prompt)

gr.Interface(fn=route_with_modal, inputs="text", outputs="text").launch()
