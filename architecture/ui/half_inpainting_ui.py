from .utils import *
import os

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sdxl_checkpoint_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SDXL' in x]
sdxl_checkpoint_list.sort()

def build_half_inpainting_ui(image, mask, prompt, ip_addr) :
    checkpoint = gr.Dropdown(sdxl_checkpoint_list, label="Select SDXL checkpoint", value = sdxl_checkpoint_list[0])

    extra_inputs = [gr.Textbox(ip_addr, visible=False)]

    base_inputs = [checkpoint, image, mask, prompt] + generate_base_checkpoint_ui()

    with gr.Group():
        with gr.Row():
            with gr.Column():
                canny_inputs = generate_controlnet_ui('SDXL', 'Canny', checkpoint_root)
                pose_inputs = generate_controlnet_ui('SDXL', 'Pose', checkpoint_root)
            with gr.Column():
                normal_inputs = generate_controlnet_ui('SDXL', 'Normal', checkpoint_root)
                depth_inputs = generate_controlnet_ui('SDXL', 'Depth', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + depth_inputs + normal_inputs + pose_inputs + extra_inputs, generate
