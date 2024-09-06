import gradio as gr
from utils import resolution_list
import os
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

flux_unet_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'unet')) if 'FLUX' in x]
flux_vae_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'vae')) if 'FLUX' in x]
flux_clip_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'clip')) if x.endswith('safetensors')]

flux_controlnet_canny_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'controlnet')) if 'FLUX_Canny' in x]
flux_controlnet_depth_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'controlnet')) if 'FLUX_Depth' in x]
flux_lora_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'loras')) if 'FLUX' in x]

flux_unet_list.sort()
flux_vae_list.sort()
flux_clip_list.sort()
flux_controlnet_canny_list.sort()
flux_controlnet_depth_list.sort()
flux_lora_list.sort()
flux_lora_list = ['None'] + flux_lora_list

def build_flux_ui(image, mask, prompt, ip_addr) :
    unet_name = gr.Dropdown(flux_unet_list, label="Select flux unet model", value = flux_unet_list[0])
    vae_name = gr.Dropdown(flux_vae_list, label="Select flux vae model", value = flux_vae_list[0])
    with gr.Row():
        with gr.Column():
            clip_1 = gr.Dropdown(flux_clip_list, label="Select flux clip first model", value=flux_clip_list[2])
        with gr.Column():
            clip_2 = gr.Dropdown(flux_clip_list, label="Select flux clip second model", value=flux_clip_list[1])

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
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=3.5, step=0.1)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=1.0, step=0.01)
        resolution = gr.Dropdown(resolution_list, label='Select Resolution (H, W)', value=resolution_list[11])
        seed = gr.Number(label='Seed', value= -1)

    base_inputs = [
        unet_name,
        vae_name,
        clip_1,
        clip_2,
        image,
        mask,
        prompt,
        resolution,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
    ]

    with gr.Accordion("ControlNet[Canny]", open=False) :
        canny_enable = gr.Checkbox(label="Enable Canny")
        canny_model_name = gr.Dropdown(flux_controlnet_canny_list, label="Select Canny model", value = flux_controlnet_canny_list[0])
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

    with gr.Accordion("LoRA", open=False):
        lora_enable = gr.Checkbox(label="Enable LoRA")
        with gr.Row():
            with gr.Group():
                lora_model_name_1 = gr.Dropdown(flux_lora_list, label="Select LoRA model", value=flux_lora_list[0])
                strength_model_1 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_1 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)
            with gr.Group():
                lora_model_name_2 = gr.Dropdown(flux_lora_list, label="Select LoRA model", value=flux_lora_list[0])
                strength_model_2 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_2 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)
            with gr.Group():
                lora_model_name_3 = gr.Dropdown(flux_lora_list, label="Select LoRA model", value=flux_lora_list[0])
                strength_model_3 = gr.Slider(label="Strength UNet", minimum=-100, maximum=100, value=1, step=0.05)
                strength_clip_3 = gr.Slider(label="Strength CLIP", minimum=-100, maximum=100, value=1, step=0.05)

    lora_inputs = [
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

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + lora_inputs + extra_inputs , generate