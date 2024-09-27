import gradio as gr
import os

def build_gemini_ui() :
    with gr.Row():
        image = gr.Image(sources='upload', type="numpy", label="Image for caption")
    with gr.Row():
        query = gr.Textbox(label="Query")
    with gr.Row() :
        result = gr.Textbox(label="Result")
    with gr.Row():
        run_gemini = gr.Button("Run Gemini!")
    gemini_inputs = [query,
                     image,
                     gr.Text('', visible=False),
                     gr.Text('', visible=False),
                     gr.Text('', visible=False),
                     ]
    return gemini_inputs, [result, run_gemini]