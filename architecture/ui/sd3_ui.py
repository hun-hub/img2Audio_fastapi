from .utils import *
import os

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sd3_checkpoint_list = ['None'] + [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SD3' in x]
sd3_checkpoint_list.sort()

controlnet_canny_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'controlnet')) if 'SD3_Canny' in x]

def build_sd3_ui(image, mask, prompt, ip_addr) :
    checkpoint = gr.Dropdown(sd3_checkpoint_list, label="Select SD3 checkpoint", value = sd3_checkpoint_list[0])

    extra_inputs = generate_gen_type_ui(ip_addr)

    base_inputs = [checkpoint, image, mask, prompt] + generate_base_checkpoint_ui(steps=20, cfg = 3, denoise = 1.0)

    with gr.Group():
        with gr.Row():
            with gr.Column():
                canny_inputs = generate_controlnet_ui('SD3', 'Canny', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + extra_inputs , generate