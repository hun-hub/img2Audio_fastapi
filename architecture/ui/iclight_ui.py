from types import NoneType

import gradio as gr
from functions.iclight.utils import generate_gradation, expand_mask
import os
import numpy as np
from PIL import Image
from cgen_utils.image_process import resize_image_for_sd

checkpoint_root = os.getenv('CHECKPOINT_ROOT')
sd15_checkpoint_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'checkpoints')) if 'SD15' in x]
iclight_model_list = [x for x in os.listdir(os.path.join(checkpoint_root, 'unet')) if 'SD15_iclight' in x]

sd15_checkpoint_list.sort()
iclight_model_list.sort()

light_condition_type_list = ['Canvas', 'Pre-Defined', 'Upload']
predefined_condition_type_list = ['None', 'Left', 'Right', 'Top', 'Bottom']
image_blendig_mode = [
                    "add",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light"
                ]

def build_iclight_ui(image, mask, prompt, ip_addr) :
    checkpoint = gr.Dropdown(sd15_checkpoint_list, label="Select SD15 checkpoint", value = sd15_checkpoint_list[0])
    iclight_model_name = gr.Dropdown(iclight_model_list, label="Select IC-Light model", value=iclight_model_list[0])

    with gr.Row():
        with gr.Column():
            light_condition_canvas = gr.ImageMask(type="numpy", label="Draw Light to Image", )
            light_condition_predefined = gr.Image(sources='upload', type="numpy", label="Selected Pre-defined Light Condition", visible=False)
            light_condition_upload = gr.Image(sources='upload', type="numpy", label="Upload Light Condition", visible=False, interactive=True)

            with gr.Group():
                light_condition_type = gr.Radio(choices=[light_type for light_type in light_condition_type_list],
                                                value=light_condition_type_list[0],
                                                label="Select Light condition", type='value')
                predefined_light_condition_type = gr.Radio(choices=[light_type for light_type in predefined_condition_type_list],
                                                           value=predefined_condition_type_list[0],
                                                           label="Select Pre Defined condition type", type='value',
                                                           visible=False)
        with gr.Column():
            light_condition = gr.Image(sources='upload', type="numpy", label="Light Condition", interactive=False, visible=False)
            light_condition_strength_applied = gr.Image(sources='upload', type="numpy", label="Light Condition", interactive=False, visible=True)

    with gr.Group():
        keep_background = gr.Radio(choices=['True', 'False'],
                                   value='True',
                                   label="Keep Background", type='value')

    with gr.Accordion("IC-Light Options", open=False):
        light_condition_strength = gr.Slider(label="Light Strength", minimum=0, maximum=1.0, value=0.5, step=0.05)
        with gr.Group():
            with gr.Row():
                blending_mode_1 = gr.Radio(choices=image_blendig_mode,
                                           value='color',
                                           label="Blending mode 1",
                                           type='value',
                                           visible=True)
                blending_percentage_1 = gr.Slider(label="Blending Percentage 1",
                                                  minimum=0,
                                                  maximum=1.0,
                                                  value=0.1,
                                                  step=0.05,
                                                  visible=True)

            with gr.Row():
                blending_mode_2 = gr.Radio(choices=image_blendig_mode,
                                           value='hue',
                                           label="Blending mode 2",
                                           type='value',
                                           visible=True)
                blending_percentage_2 = gr.Slider(label="Blending Percentage 2",
                                                  minimum=0,
                                                  maximum=1.0,
                                                  value=0.2,
                                                  step=0.05,
                                                  visible=True)
        with gr.Group():
            remap_min_value = gr.Slider(label="Remap Image Min Value",
                                        minimum=-10.0,
                                        maximum=10.0,
                                        value=-0.15,
                                        step=0.05,
                                        visible=True)
            remap_max_value = gr.Slider(label="Remap Image Max Value",
                                        minimum=-10.0,
                                        maximum=10.0,
                                        value=1.14,
                                        step=0.05,
                                        visible=True)

    with gr.Accordion("Options", open=False):
        num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=30, step=1)
        guidance_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, value=1.5, step=0.1)
        denoising_strength = gr.Slider(label="Denoising strength", minimum=0.1, maximum=1.0, value=0.7, step=0.01)
        seed = gr.Number(label='Seed', value= -1)

    with gr.Row() :
        generate = gr.Button("Generate!")

    # Light Canvas Process
    image.change(fn=update_light_conditions_by_image, inputs=[image, light_condition_type, light_condition_upload, predefined_light_condition_type], outputs=[light_condition_canvas, light_condition])

    light_condition_canvas.change(fn=light_condition_update_by_canvas, inputs=[light_condition_canvas, light_condition_type], outputs = [light_condition])
    # Light Upload Process
    light_condition_upload.change(fn=light_condition_update_by_upload, inputs=[image, light_condition_upload], outputs=[light_condition])
    # Light Pre-defined Process
    predefined_light_condition_type.change(fn=light_condition_update_by_predefined, inputs=[gr.Text('', visible=False), predefined_light_condition_type], outputs = [light_condition_predefined])
    predefined_light_condition_type.change(fn=light_condition_update_by_predefined, inputs=[image, predefined_light_condition_type], outputs = [light_condition])
    # light codition 에 따라 visible 설정 + predefined visible 까지
    light_condition_type.change(fn=change_canvas_type,
                                inputs=[light_condition_type, image, light_condition_canvas, predefined_light_condition_type, light_condition_upload],
                                outputs = [light_condition_canvas, light_condition_predefined, light_condition_upload, predefined_light_condition_type, light_condition])

    keep_background_options=[blending_mode_1, blending_percentage_1, blending_mode_2, blending_percentage_2, remap_min_value, remap_max_value]
    keep_background.change(fn=update_ui_by_keep_background, inputs = [keep_background], outputs = keep_background_options)
    # Light Strength 적용
    light_condition.change(fn=apply_light_condition, inputs=[light_condition_strength, light_condition], outputs=[light_condition_strength_applied])
    light_condition_strength.change(fn=apply_light_condition, inputs=[light_condition_strength, light_condition], outputs=[light_condition_strength_applied])

    base_inputs = [
        checkpoint,
        image,
        mask,
        prompt,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
    ]

    iclight_inputs = [iclight_model_name,
                      light_condition,
                      light_condition_strength,
                      keep_background,
                      blending_mode_1,
                      blending_percentage_1,
                      blending_mode_2,
                      blending_percentage_2,
                      remap_min_value,
                      remap_max_value
                      ]

    extra_inputs = [
        gr.Text(ip_addr, visible=False)
    ]

    return base_inputs + iclight_inputs + extra_inputs, generate


def update_light_conditions_by_image(image_array, condition_type, upload_array, predefined_type):
    if condition_type == 'Canvas' :
        return gr.update(value=image_array), None
    elif condition_type == 'Pre-Defined' :
        return gr.update(value=image_array), light_condition_update_by_predefined(image_array, predefined_type)
    elif condition_type == 'Upload' :
        return gr.update(value=image_array), light_condition_update_by_upload(image_array, upload_array)
    else :
        return None, None

def change_canvas_type(condition_type, image_array, canvas, predefined_type, upload):
    if condition_type == 'Canvas' :
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), light_condition_update_by_canvas(canvas, condition_type)
    elif condition_type == 'Pre-Defined' :
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), light_condition_update_by_predefined(image_array, predefined_type)
    elif condition_type == 'Upload' :
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), light_condition_update_by_upload(image_array, upload)


def light_condition_update_by_canvas(canvas, condition_type) :
    if condition_type != 'Canvas' : return gr.update()
    background, layers, composite = canvas.values()
    if background[:, :, :3].sum() == 0 : return None
    light_mask = layers[0][:, :, 3]
    return gr.update(value=light_mask)

def light_condition_update_by_upload(image_array, upload_array) :
    if isinstance(upload_array, NoneType) or isinstance(image_array, NoneType) : return None
    height, width, _ = image_array.shape
    upload_light_gradation = Image.fromarray(upload_array).resize((width, height))
    upload_light_gradation_array = np.array(upload_light_gradation)
    return gr.update(value=upload_light_gradation_array)

def light_condition_update_by_predefined(image_array, predefined_type) :
    light_gradation = generate_gradation(predefined_type, image_array)
    return gr.update(value=light_gradation)

def update_ui_by_keep_background(keep_background):
    visible = True if keep_background == 'True' else False
    return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)

# TODO: 원래는 type 마다 따로 연산해서 원래 light_gradation 가지고 있어야함.
def apply_light_condition(strength, light_array) :
    if isinstance(light_array, NoneType) : return None

    strength = int((strength - 0.5) * 200)
    light_image = Image.fromarray(light_array)
    light_image_resized = resize_image_for_sd(light_image)
    light_image_resized_array = np.array(light_image_resized)
    light_image_resized_array = expand_mask(light_image_resized_array, strength)
    return gr.update(value=light_image_resized_array)