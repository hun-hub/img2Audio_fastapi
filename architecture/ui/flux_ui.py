import gradio as gr
from utils import resolution_list
import os

flux_unet_list = [x for x in os.listdir('/checkpoints/unet') if 'FLUX' in x]
flux_vae_list = [x for x in os.listdir('/checkpoints/vae') if 'FLUX' in x]
flux_controlnet_canny_list = [x for x in os.listdir('/checkpoints/vae') if 'FLUX' in x]

flux_unet_list.sort()
flux_vae_list.sort()

def build_flux_ui(image, mask, prompt, ip_addr) :
    unet_name = gr.Dropdown(flux_unet_list, label="Select flux unet model", value = flux_unet_list[0])
    vae_name = gr.Dropdown(flux_vae_list, label="Select flux vae model", value = flux_vae_list[0])

    with gr.Row() :
        gen_type = gr.Radio(
            choices=['t2i', 'i2i', 'inpaint'],
            value='t2i',
            label="Generation Type",
            type='value'
        )

    extra_inputs = [
        gen_type,
        gr.Text(ip_addr, visible=False)
    ]

    with gr.Accordion("Options", open=False):
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=28, step=1)
        timestep_to_start_cfg  = gr.Slider(label="Time step to start CFG", minimum=10, maximum=100, value=0, step=1)
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=3, step=0.1)
        i2i_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=1.0, step=0.01)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=1.0, step=0.01)
        resolution = gr.Dropdown(resolution_list, label='Select Resolution (H, W)', value=resolution_list[11])
        seed = gr.Number(label='Seed', value= -1)

    base_inputs = [
        unet_name,
        vae_name,
        image,
        mask,
        prompt,

        num_inference_steps,
        timestep_to_start_cfg,
        guidance_scale,
        i2i_strength,
        denoising_strength,
        resolution,
        seed,
    ]

    with gr.Accordion("ControlNet[Canny]", open=False) :
        canny_enable = gr.Checkbox(label="Enable Canny")
        canny_model_name = gr.Dropdown(controlnet_canny_list, label="Select Canny model", value = controlnet_canny_list[0])
        canny_image = gr.Image(sources='upload', type="numpy", label="Control(Canny) Image, Auto Transform")
        canny_control_weight = gr.Slider(label="Canny Control Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        canny_start = gr.Slider(label="Canny Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        canny_end = gr.Slider(label="Canny End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    canny_inputs = [
        canny_enable,
        canny_model_name,
        canny_image,
        canny_control_weight,
        canny_start,
        canny_end,
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + extra_inputs , generate