from .utils import *
import os

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sd15_checkpoint_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SD15' in x]
sd15_checkpoint_list.sort()

def build_sd15_ui(image, mask, prompt, ip_addr) :
    checkpoint = gr.Dropdown(sd15_checkpoint_list, label="Select SD15 checkpoint", value = sd15_checkpoint_list[0])

    extra_inputs = generate_gen_type_ui(ip_addr)

    base_inputs = [checkpoint, image, mask, prompt] + generate_base_checkpoint_ui()

    with gr.Group():
        with gr.Row():
            with gr.Column():
                canny_inputs = generate_controlnet_ui('SD15', 'Canny', checkpoint_root)
                pose_inputs = generate_controlnet_ui('SD15', 'Pose', checkpoint_root)
            with gr.Column():
                normal_inputs = generate_controlnet_ui('SD15', 'Normal', checkpoint_root)
                depth_inputs = generate_controlnet_ui('SD15', 'Depth', checkpoint_root)
            with gr.Column():
                inpaint_inputs = generate_controlnet_ui('SD15', 'Inpaint', checkpoint_root)

    ipadapter_inputs = generate_ipadapter_ui('SD15', checkpoint_root)

    lora_inputs = generate_lora_ui('SD15', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + canny_inputs + inpaint_inputs + depth_inputs + normal_inputs + pose_inputs + ipadapter_inputs + lora_inputs + extra_inputs, generate