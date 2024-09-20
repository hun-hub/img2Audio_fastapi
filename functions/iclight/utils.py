import torch
from utils import set_comfyui_packages
from utils.loader import get_function_from_comfyui
import requests
from utils.handler import handle_response
from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 convert_base64_to_image_tensor
                                 )
from utils.comfyui import (make_canny,
                           apply_controlnet,
                           apply_ipadapter,
                           make_image_batch)
from utils.loader import load_clip_vision
from types import NoneType
from PIL import Image
import numpy as np
import scipy
import os
# set_comfyui_packages()

def generate_gradation(light_condition, image_array):
    if isinstance(image_array, np.ndarray):
        height, width, _ = image_array.shape
    else:
        height, width = 1024, 1024

    def exponential_gradient(start, stop, num):
        linear_gradient = np.linspace(start, stop, num)
        return linear_gradient * 255

    if light_condition == 'Left':
        gradient = exponential_gradient(1, 0, width)
        image = np.tile(gradient, (height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif light_condition == 'Right':
        gradient = exponential_gradient(0, 1, width)
        image = np.tile(gradient, (height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif light_condition == 'Top':
        gradient = exponential_gradient(1, 0, height)[:, None]
        image = np.tile(gradient, (1, width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif light_condition == 'Bottom':
        gradient = exponential_gradient(0, 1, height)[:, None]
        image = np.tile(gradient, (1, width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else :
        input_bg = None

    return input_bg


def expand_mask(mask: np.ndarray, expand:int, tapered_corners:bool = True):
    # mask shape: (H, W, 3)
    mask = torch.Tensor(mask[:, :, 0]).squeeze().unsqueeze(0) / 255

    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    mask_expanded_array = torch.stack(out, dim=0).squeeze().numpy() * 255
    return np.stack([mask_expanded_array] * 3, axis=-1).astype(np.uint8)


def load_and_apply_iclight(unet, iclight_model_path):
    module_path = 'ComfyUI/custom_nodes/ComfyUI-IC-Light'
    func_name = 'nodes.LoadAndApplyICLightUnet'
    iclight_applier = get_function_from_comfyui(module_path, func_name)
    unet = iclight_applier().load(unet, iclight_model_path)
    return unet[0]

def load_iclight_condition_applier() :
    module_path = 'ComfyUI/custom_nodes/ComfyUI-IC-Light'
    func_name = 'nodes.ICLightConditioning'
    iclight_conditioning = get_function_from_comfyui(module_path, func_name)
    return iclight_conditioning()


def construct_condition(unet,
                        iclight_model,
                        positive,
                        negative,
                        init_noise
                        ):
    # iclight unet apply
    unet = load_and_apply_iclight(unet, iclight_model)
    # ic light conditioning
    iclight_conditioning = load_iclight_condition_applier()
    positive, negative, _ = iclight_conditioning.encode(positive,
                                                        negative,
                                                        None,
                                                        init_noise,
                                                        multiplier=0.156)

    return unet, positive, negative

def frequency_separate(image_tensor, blur_radius=3) :
    module_path = 'ComfyUI/custom_nodes/comfyUI_FrequencySeparation_RGB-HSV'
    func_name = 'frequency_separation.FrequencySeparation'
    seperator = get_function_from_comfyui(module_path, func_name)

    high_freq, low_freq = seperator().separate(image_tensor, blur_radius)
    return high_freq, low_freq

def frequency_combination(high_freq_image_tensor, low_freq_image_tensor) :
    module_path = 'ComfyUI/custom_nodes/comfyUI_FrequencySeparation_RGB-HSV'
    func_name = 'frequency_combination.FrequencyCombination'
    combinator = get_function_from_comfyui(module_path, func_name)
    combined = combinator().combine(high_freq_image_tensor, low_freq_image_tensor)
    return combined[0]

def was_image_blend(image_tensor_1, image_tensor_2, mode, blend_percentage) :
    module_path = 'ComfyUI/custom_nodes/was-node-suite-comfyui'
    func_name = 'WAS_Node_Suite.WAS_Image_Blending_Mode'
    blender = get_function_from_comfyui(module_path, func_name)
    blended_image = blender().image_blending_mode(image_tensor_1, image_tensor_2, mode, blend_percentage)
    return blended_image[0]

def remap_image(image_tensor, min=-0.15, max=1.14) :
    image_tensor = image_tensor.to(torch.float32)
    image_tensor = min + image_tensor * (max - min)
    image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0)
    return image_tensor

def sned_iclight_request_to_api(
        checkpoint,
        image,
        mask,
        prompt,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,

        iclight_model_name,
        light_condition,
        light_condition_strength,
        keep_background,
        blending_mode_1,
        blending_percentage_1,
        blending_mode_2,
        blending_percentage_2,
        remap_min_value,
        remap_max_value,

        ip_addr
) :
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image))
        image = convert_image_to_base64(image)
    if not isinstance(mask, NoneType):
        mask = resize_image_for_sd(Image.fromarray(mask), is_mask=True)
        mask = convert_image_to_base64(mask)
    if not isinstance(light_condition, NoneType):
        light_condition = resize_image_for_sd(Image.fromarray(light_condition))
        light_condition = convert_image_to_base64(light_condition)

    request_body = {
        'checkpoint': checkpoint,
        'init_image': image,
        'mask': mask,
        "prompt_positive": prompt,
        "prompt_negative": "",
        'steps': num_inference_steps,
        'cfg': guidance_scale,
        'denoise': denoising_strength,
        'seed': seed,

        'iclight_model': iclight_model_name,
        'light_condition': light_condition,
        'light_strength': light_condition_strength,
        'keep_background': True if keep_background == 'True' else False,
        'blending_mode_1': blending_mode_1,
        'blending_percentage_1': blending_percentage_1,
        'blending_mode_2': blending_mode_2,
        'blending_percentage_2': blending_percentage_2,
        'remap_min_value': remap_min_value,
        'remap_max_value': remap_max_value,

        'controlnet_requests': [],
    }

    url = f"http://{ip_addr}/iclight/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image