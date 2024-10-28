from .utils import *
import os

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sdxl_checkpoint_list = ['None'] + [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SDXL' in x]
sdxl_checkpoint_list.sort()

def build_sdxl_ui(image, mask, prompt, ip_addr) :
    checkpoint = gr.Dropdown(sdxl_checkpoint_list, label="Select SDXL checkpoint", value = sdxl_checkpoint_list[0])

    extra_inputs = generate_gen_type_ui(ip_addr)

    base_inputs = [checkpoint, image, mask, prompt] + generate_base_checkpoint_ui()
    refiner_inputs = generate_refiner_checkpoint_ui(sdxl_checkpoint_list)
    with gr.Group():
        with gr.Row():
            with gr.Column():
                canny_inputs = generate_controlnet_ui('SDXL', 'Canny', checkpoint_root)
                pose_inputs = generate_controlnet_ui('SDXL', 'Pose', checkpoint_root)
            with gr.Column():
                normal_inputs = generate_controlnet_ui('SDXL', 'Normal', checkpoint_root)
                depth_inputs = generate_controlnet_ui('SDXL', 'Depth', checkpoint_root)
            with gr.Column():
                inpaint_inputs = generate_controlnet_ui('SDXL', 'Inpaint', checkpoint_root)

    ipadapter_inputs = generate_ipadapter_ui('SDXL', checkpoint_root)

    lora_inputs = generate_lora_ui('SDXL', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    return base_inputs + refiner_inputs + canny_inputs + inpaint_inputs + depth_inputs + normal_inputs + pose_inputs + ipadapter_inputs + lora_inputs + extra_inputs, generate
