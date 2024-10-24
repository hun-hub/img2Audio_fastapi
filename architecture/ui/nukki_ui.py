import gradio as gr
import os
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

nukki_model_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'BiRefNet'))]
nukki_model_list.sort()

def build_nukki_ui(image, ip_addr) :
    model_name = gr.Dropdown(nukki_model_list, label="Select Nukki model", value = nukki_model_list[0])
    with gr.Row():
        with gr.Column():
            nukki = gr.ImageMask(sources='upload', type="numpy", label="Edit Nukki")
            edit_mode = gr.Radio(choices=['add_white', 'add_black'],
                                 value='add_white',
                                 label="Select edit mode",
                                 type='value')
    base_inputs = [
        model_name,
        image,
        nukki,
        edit_mode
    ]

    extra_inputs = [
        gr.Text(ip_addr, visible=False)
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + extra_inputs, generate
