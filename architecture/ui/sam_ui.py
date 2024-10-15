import gradio as gr
import os
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

def build_sam_ui(image, prompt, ip_addr) :
    threshold = gr.Slider(label="Detect Threshold", minimum=0, maximum=1.0, value=0.2, step=0.05)

    inputs = [image, prompt, threshold]

    extra_inputs = [
        gr.Text(ip_addr, visible=False)
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return inputs + extra_inputs, generate
