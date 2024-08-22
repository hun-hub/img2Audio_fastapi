import gradio as gr
import os

def build_gemini_ui(ip_addr) :
    with gr.Row():
        image = gr.Image(sources='upload', type="numpy", label="Image for caption")
    with gr.Row():
        query = gr.Textbox(label="Query")
    with gr.Row():
        user_prompt = gr.Textbox(label="User Prompt")
    with gr.Row() :
        result = gr.Textbox(label="Result")
    with gr.Row():
        run_gemini = gr.Button("Run Gemini!")
    return [image, query, user_prompt, gr.Text(ip_addr, visible=False)], [result, run_gemini]