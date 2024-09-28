import gradio as gr
from cgen_utils import resolution_list
import os

def generate_gen_type_ui(ip_addr) :
    with gr.Row() :
        gen_type = gr.Radio(
            choices=['t2i', 'i2i', 'inpaint'],
            value='t2i',
            label="Generation Type",
            type='value'
        )

    inputs = [
        gen_type,
        gr.Textbox(ip_addr, visible=False)
    ]
    return inputs

def generate_base_checkpoint_ui(steps=20, cfg = 7, denoise = 1.0) :
    with gr.Accordion("Options", open=False):
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=steps, step=1)
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=cfg, step=0.1)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=denoise, step=0.01)
        resolution = gr.Dropdown(resolution_list, label='Select Resolution (H, W)', value=resolution_list[11])
        seed = gr.Number(label='Seed', value= -1)

    inputs = [
        resolution,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
    ]
    return inputs

def generate_refiner_checkpoint_ui(checkpoint_list) :
    with gr.Accordion("Refiner", open=False):
        enable = gr.Checkbox(label="Enable Refiner")
        name = gr.Dropdown(checkpoint_list, label="Select Refiner model", value=checkpoint_list[0])
        switch = gr.Slider(label="Swich", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    inputs = [
        enable,
        name,
        switch,
    ]

    return inputs

def generate_controlnet_ui(sd_type: str, controlnet_type: str, checkpoint_root: str) :
    controlnet_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'controlnet')) if f'{sd_type}_{controlnet_type}' in x]
    controlnet_list.sort()
    if len(controlnet_list) == 0 :
        controlnet_list = ['None']

    preprocessor_type_list = ['canny', 'lineart', 'openpose', 'normalmap_bae', 'normalmap_midas', 'depth_midas', 'depth', 'depth_zoe']

    with gr.Accordion(f"ControlNet[{controlnet_type}]", open=False) :
        enable = gr.Checkbox(label=f"Enable {controlnet_type}")
        model_name = gr.Dropdown(controlnet_list, label=f"Select {controlnet_type} model", value = controlnet_list[0])
        if controlnet_type == 'Inpaint' :
            with gr.Row():
                with gr.Column():
                    image = gr.Image(sources='upload', type="numpy", label=f"Control({controlnet_type}) Image")
                with gr.Column():
                    mask = gr.Image(sources='upload', type="numpy", label=f"Control({controlnet_type}) Mask")
        else :
            image = gr.Image(sources='upload', type="numpy", label=f"Control({controlnet_type}) Image, Auto Transform")
        preprocessor_type = gr.Dropdown(preprocessor_type_list, label=f"Select {controlnet_type} pre-processor type", value=preprocessor_type_list[0])
        control_weight = gr.Slider(label=f"Canny {controlnet_type} Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        start = gr.Slider(label=f"{controlnet_type} Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        end = gr.Slider(label=f"{controlnet_type} End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    inputs = [
        enable,
        model_name,
        image,
        preprocessor_type,
        control_weight,
        start,
        end,
    ]
    if controlnet_type == 'Inpaint':
        inputs.insert(3, mask)

    return inputs

def generate_ipadapter_ui(sd_type: str, checkpoint_root: str) :
    ipadapter_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'ipadapter')) if sd_type in x]

    with gr.Accordion("IP-Adapter", open=False):
        ipadapter_enable = gr.Checkbox(label="Enable IP-Adapter")
        ipadapter_model_name = gr.Dropdown(ipadapter_list, label="Select IP-Adapter model",
                                       value=ipadapter_list[0])
        ipadapter_images = gr.Gallery(type='numpy',
                                      label='IP-Adapter Images (Recommand 3 images!!)',
                                      columns = 3,
                                      interactive=True)
        ipadapter_weight = gr.Slider(label="IP-Adapter Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        ipadapter_start = gr.Slider(label="IP-Adapter Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        ipadapter_end = gr.Slider(label="IP-Adapter End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)

    inputs = [
        ipadapter_enable,
        ipadapter_model_name,
        ipadapter_images,
        ipadapter_weight,
        ipadapter_start,
        ipadapter_end,
    ]
    return inputs

def generate_lora_ui(sd_type: str, checkpoint_root: str) :
    lora_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'loras')) if sd_type in x]
    lora_list.sort()
    lora_list = ['None'] + lora_list

    with gr.Accordion("LoRA", open=False):
        lora_enable = gr.Checkbox(label="Enable LoRA")
        with gr.Row():
            with gr.Group():
                lora_model_name_1 = gr.Dropdown(lora_list, label="Select LoRA model", value=lora_list[0])
                strength_model_1 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_1 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)
            with gr.Group():
                lora_model_name_2 = gr.Dropdown(lora_list, label="Select LoRA model", value=lora_list[0])
                strength_model_2 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_2 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)
            with gr.Group():
                lora_model_name_3 = gr.Dropdown(lora_list, label="Select LoRA model", value=lora_list[0])
                strength_model_3 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_3 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)

    inputs = [
        lora_enable,
        lora_model_name_1,
        strength_model_1,
        strength_clip_1,
        lora_model_name_2,
        strength_model_2,
        strength_clip_2,
        lora_model_name_3,
        strength_model_3,
        strength_clip_3,
    ]
    return inputs