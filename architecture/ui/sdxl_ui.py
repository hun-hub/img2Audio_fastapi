import gradio as gr
from utils import resolution_list
import os

sdxl_model_list = [x for x in os.listdir('/checkpoints/sdxl_light') if 'SDXL' in x]
controlnet_canny_list = [x for x in os.listdir('/checkpoints/controlnet') if 'SDXL_Canny' in x]
controlnet_inpaint_list = [x for x in os.listdir('/checkpoints/controlnet') if 'SDXL_Inpaint' in x]
ipadapter_list = [x for x in os.listdir('/checkpoints/ipadapter') if 'SDXL' in x]

sdxl_model_list.sort()
controlnet_canny_list.sort()
controlnet_inpaint_list.sort()
ipadapter_list.sort()

def build_sdxl_ui(image, mask, prompt, ip_addr) :
    model_name = gr.Dropdown(sdxl_model_list, label="Select SDXL model", value = sdxl_model_list[0])

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
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=20, step=1)
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=7, step=0.1)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=1.0, step=0.01)
        resolution = gr.Dropdown(resolution_list, label='Select Resolution (H, W)', value=resolution_list[11])
        seed = gr.Number(label='Seed', value= -1)

    base_inputs = [
        model_name,
        image,
        mask,
        prompt,
        resolution,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
    ]

    with gr.Accordion("Refiner", open=False):
        refiner_enable = gr.Checkbox(label="Enable Refiner")
        refiner_name = gr.Dropdown(sdxl_model_list, label="Select Refiner model", value=sdxl_model_list[0])
        refine_switch = gr.Slider(label="Swich", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    refiner_inputs = [
        refiner_enable,
        refiner_name,
        refine_switch,
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

    with gr.Accordion("ControlNet[Inpaint]", open=False):
        inpaint_enable = gr.Checkbox(label="Enable Inpaint")
        inpaint_model_name = gr.Dropdown(controlnet_inpaint_list, label="Select Canny model",
                                       value=controlnet_inpaint_list[0])
        with gr.Row() :
            with gr.Column():
                inpaint_image = gr.Image(sources='upload', type="numpy", label="Control(Inpaint) Image")
            with gr.Column():
                inpaint_mask = gr.Image(sources='upload', type="numpy", label="Control(Inpaint) Mask")
        inpaint_control_weight = gr.Slider(label="Inpaint Control Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        inpaint_start = gr.Slider(label="Inpaint Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        inpaint_end = gr.Slider(label="Inpaint End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    inpaint_inputs = [
        inpaint_enable,
        inpaint_model_name,
        inpaint_image,
        inpaint_mask,
        inpaint_control_weight,
        inpaint_start,
        inpaint_end,
    ]

    with gr.Accordion("IP-Adapter", open=False):
        ipadapter_enable = gr.Checkbox(label="Enable IP-Adapter")
        ipadapter_model_name = gr.Dropdown(ipadapter_list, label="Select IP-Adapter model",
                                       value=ipadapter_list[0])
        # ipadapter_image = gr.Image(sources='upload', type="numpy", label="IP-Adapter Image")
        ipadapter_images = gr.Gallery(type='numpy',
                                      label='IP-Adapter Images (Recommand 3 images!!)',
                                      columns = 3,
                                      interactive=True)
        ipadapter_weight = gr.Slider(label="IP-Adapter Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        ipadapter_start = gr.Slider(label="IP-Adapter Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        ipadapter_end = gr.Slider(label="IP-Adapter End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    ipadapter_inputs = [
        ipadapter_enable,
        ipadapter_model_name,
        ipadapter_images,
        ipadapter_weight,
        ipadapter_start,
        ipadapter_end,
    ]

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + refiner_inputs + canny_inputs + inpaint_inputs + ipadapter_inputs + extra_inputs, generate
