from .utils import *
import os
checkpoint_root = os.getenv('CHECKPOINT_ROOT')

flux_unet_list = ['None'] + [x for x in os.listdir(os.path.join(checkpoint_root, 'unet')) if 'FLUX' in x]
flux_vae_list = ['None'] + [x for x in os.listdir(os.path.join(checkpoint_root, 'vae')) if 'FLUX' in x]
flux_clip_list = ['None'] + [x for x in os.listdir(os.path.join(checkpoint_root, 'clip')) if x.endswith('safetensors')]
flux_unet_list.sort()
flux_vae_list.sort()
flux_clip_list.sort()

def build_flux_ui(image, mask, prompt, ip_addr) :
    unet_name = gr.Dropdown(flux_unet_list, label="Select FLUX unet model", value = flux_unet_list[0])
    vae_name = gr.Dropdown(flux_vae_list, label="Select FLUX vae model", value = flux_vae_list[0])
    with gr.Row():
        with gr.Column():
            clip_1 = gr.Dropdown(flux_clip_list, label="Select FLUX clip first model", value=flux_clip_list[2])
        with gr.Column():
            clip_2 = gr.Dropdown(flux_clip_list, label="Select FLUX clip second model", value=flux_clip_list[1])

    extra_inputs = generate_gen_type_ui(ip_addr)

    base_inputs = [unet_name, vae_name, clip_1, clip_2, image, mask, prompt] + generate_base_checkpoint_ui(cfg=3.5)

    with gr.Group():
        with gr.Row():
            with gr.Column():
                canny_inputs = generate_controlnet_ui('FLUX', 'Canny', checkpoint_root)
            with gr.Column():
                depth_inputs = generate_controlnet_ui('FLUX', 'Depth', checkpoint_root)

    lora_inputs = generate_lora_ui('FLUX', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + depth_inputs + lora_inputs + extra_inputs , generate