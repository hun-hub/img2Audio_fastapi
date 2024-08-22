import gradio as gr
from utils import resolution_list
import os

upscale_model_list = [x for x in os.listdir('/checkpoints/upscale_models')]
upscale_model_list.sort()
upscale_methods = ['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']

def build_upscale_ui( ip_addr) :
    model_name = gr.Dropdown(upscale_model_list, label="Select Upscale model", value = upscale_model_list[0])
    with gr.Row():
        with gr.Column():
            image = gr.Image(sources='upload', type="numpy", label="Upload Image")
    with gr.Accordion("Options", open=False):
        method = gr.Dropdown(upscale_methods, label="Select Upscale methods", value=upscale_methods[4])
        scale = gr.Slider(label="Scale", minimum=2, maximum=4, value=2, step=0.5)

    base_inputs = [
        model_name,
        image,
        method,
        scale,
    ]

    extra_inputs = [
        gr.Text(ip_addr, visible=False)
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + extra_inputs, generate
