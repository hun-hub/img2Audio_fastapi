import gradio as gr
from utils import resolution_list
import os
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

sdxl_checkpoint_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'sdxl_light')) if 'SDXL' in x]
inpaint_model_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'inpaint')) if not x.endswith('pth') ]

sdxl_checkpoint_list.sort()
inpaint_model_list.sort()

def build_object_remove_ui(prompt, ip_addr) :
    checkpoint = gr.Dropdown(sdxl_checkpoint_list, label="Select SDXL checkpoint", value = sdxl_checkpoint_list[3])
    inpaint_model_name = gr.Dropdown(inpaint_model_list, label="Select Inpaint patch", value=inpaint_model_list[-1])
    with gr.Row():
        with gr.Column():
            inpaint_image = gr.ImageMask(sources='upload', type="numpy", label="Remove area by masking to  image")
    with gr.Accordion("Options", open=False):
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=20, step=1)
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=7, step=0.1)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=1.0, step=0.01)
        seed = gr.Number(label='Seed', value= -1)

    base_inputs = [
        checkpoint,
        prompt,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
    ]

    inpaint_inputs = [
        inpaint_model_name,
        inpaint_image,
    ]

    extra_inputs = [
        gr.Text(ip_addr, visible=False)
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + inpaint_inputs + extra_inputs, generate
