from .utils import *

def build_i2c_ui(ip_addr) :
    with gr.Row():
        image = gr.Image(sources='upload', type="numpy", label=f"Original Image")
    style_type = gr.Dropdown(['type_1', 'type_2'], label="Select Cartoon Style (type_1: candidate 4, type_2: candidate 5)", value = 'type_1')

    inputs = [image, style_type]
    extra_inputs = [gr.Text(ip_addr, visible=False)]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return inputs + extra_inputs, generate
