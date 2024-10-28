from .utils import *
import os

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sd15_checkpoint_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SD15' in x]
sd15_checkpoint_list.sort()

def detach_mask_from_canvas(canvas) :
    background, layers, composite = canvas.values()
    mask = layers[0][:, :, 3]
    if mask.sum() != 0 :
        return gr.update(value = mask)
    else :
        return None

def change_canvas(image) :
    return gr.update(value = image)

def change_mode(retouch_mode) :
    if retouch_mode == 'True' :
        return gr.update(visible = True), gr.update(visible = True), gr.update(visible = True)
    else :
        return gr.update(visible = False), gr.update(visible = False), gr.update(visible = False)

def generate_advanced_controlnet_ui(
        sd_type: str,
        controlnet_type: str,
        checkpoint_root: str,
        base_multiplier:float = 0.9,
        flip_weights:str = 'false',
        uncond_multiplier:float=1) :

    controlnet_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'controlnet')) if f'{sd_type}_{controlnet_type}' in x]
    controlnet_list.sort()
    if len(controlnet_list) == 0 :
        controlnet_list = ['None']

    preprocessor_type_list = ['canny', 'lineart', 'openpose', 'normalmap_bae', 'normalmap_midas', 'depth_midas', 'depth', 'depth_zoe', 'scribble']

    with gr.Accordion(f"ControlNet[{controlnet_type}]", open=False) :
        enable = gr.Checkbox(label=f"Enable {controlnet_type}")
        model_name = gr.Dropdown(controlnet_list, label=f"Select {controlnet_type} model", value = controlnet_list[0])
        preprocessor_type = gr.Dropdown(preprocessor_type_list, label=f"Select {controlnet_type} pre-processor type", value=preprocessor_type_list[0])
        control_weight = gr.Slider(label=f"{controlnet_type} Weight", minimum=0, maximum=3, value=0.7, step=0.05)
        start = gr.Slider(label=f"{controlnet_type} Start", minimum=0.0, maximum=1.0, value=0, step=0.05)
        end = gr.Slider(label=f"{controlnet_type} End", minimum=0.0, maximum=1.0, value=0.4, step=0.05)
        # Advanced Controlnet inputs
        base_multiplier_value = gr.Slider(label=f"{controlnet_type} Base Multiplier", minimum=0.0, maximum=1.0, value=base_multiplier, step=0.05)
        uncond_multiplier_value = gr.Slider(label=f"{controlnet_type} Uncond Multiplier", minimum=0.0, maximum=1.0, value=uncond_multiplier, step=0.05)
        flip_weights_value = gr.Radio(choices=['true', 'false'], value=flip_weights, label="Flip weights", type='value')

    inputs = [
        enable,
        model_name,
        preprocessor_type,
        control_weight,
        start,
        end,
        base_multiplier_value,
        uncond_multiplier_value,
        flip_weights_value
    ]

    return inputs

def build_bg_change_ui(image, mask, prompt, ip_addr) :
    do_retouch = gr.Radio(choices=['True', 'False'],
                          value='False',
                          label="Do Retouch",
                          type='value')

    with gr.Row():
        with gr.Column():
            image_retouch = gr.ImageMask(type="numpy", label="Masking to Retouch Area", visible=False)
        with gr.Column():
            mask_retouch = gr.Image( type="numpy", label="Masking to Retouch Area", visible=False)
        prompt_retouch = gr.Textbox(label="Retouch Prompt", visible=False)

    extra_inputs = [gr.Textbox(ip_addr, visible=False)]

    base_inputs = [image, mask, mask_retouch, prompt, prompt_retouch, do_retouch]

    # with gr.Group():
    #     with gr.Row():
    #         with gr.Column():
    #             canny_inputs = generate_advanced_controlnet_ui('SD15', 'Canny', checkpoint_root, 0.9, 'false', 1)
    #             depth_inputs = generate_advanced_controlnet_ui('SD15', 'Depth', checkpoint_root, 0.9, 'false', 1)
    #         with gr.Column():
    #             inpaint_inputs = generate_advanced_controlnet_ui('SD15', 'Inpaint', checkpoint_root, 0.87, 'true', 1)
    #             scribble_inputs = generate_advanced_controlnet_ui('SD15', 'Scribble', checkpoint_root, 0.9, 'false', 1)

    ipadapter_inputs = generate_ipadapter_ui('SD15', checkpoint_root)

    # lora_inputs = generate_lora_ui('SD15', checkpoint_root)

    with gr.Row() :
        generate = gr.Button("Generate!")

    do_retouch.change(fn=change_mode, inputs=do_retouch, outputs = [image_retouch, mask_retouch, prompt_retouch])
    image.change(fn=change_canvas, inputs=image, outputs=[image_retouch])
    image_retouch.change(fn=detach_mask_from_canvas, inputs=image_retouch, outputs=[mask_retouch])

    return base_inputs +ipadapter_inputs +  extra_inputs, generate