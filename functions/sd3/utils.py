import torch
from utils import set_comfyui_packages
from utils.loader import get_function_from_comfyui
import requests
from utils.handler import handle_response
from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 )
from types import NoneType
from PIL import Image

# set_comfyui_packages()


@torch.inference_mode()
def apply_controlnet(positive, negative, controlnet, vae, image, strength, start_percent, end_percent,) :
    from ComfyUI.comfy_extras.nodes_sd3 import ControlNetApplySD3
    controlnet_applier = ControlNetApplySD3()
    positive, negative = controlnet_applier.apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent, vae)
    return positive, negative

def model_sampling_sd3(unet, shift:float = 3) :
    from ComfyUI.comfy_extras.nodes_model_advanced import ModelSamplingSD3
    model_sampler = ModelSamplingSD3()
    unet = model_sampler.patch(unet, shift)[0]

    return unet

def get_init_noise(width, height, batch_size=1) :
    from ComfyUI.comfy_extras.nodes_sd3 import EmptySD3LatentImage
    latent_sampler = EmptySD3LatentImage()
    init_noise = latent_sampler.generate(width, height, batch_size)[0]

    return init_noise

def sned_sd3_request_to_api(
        checkpoint,
        image,
        mask,
        prompt,
        resolution,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,
        canny_enable,
        canny_model_name,
        canny_image,
        canny_control_weight,
        canny_start,
        canny_end,
        gen_type,
        ip_addr
) :
    height, width = eval(resolution)
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image))
        image = convert_image_to_base64(image)
    if not isinstance(mask, NoneType):
        mask = resize_image_for_sd(Image.fromarray(mask), is_mask=True)
        mask = convert_image_to_base64(mask)
    if not isinstance(canny_image, NoneType):
        canny_image = resize_image_for_sd(Image.fromarray(canny_image))
        canny_image = convert_image_to_base64(canny_image)

    request_body = {
        'checkpoint': checkpoint,
        'init_image': image,
        'mask': mask,
        "prompt_positive": prompt,
        "prompt_negative": "",
        'width': width,
        'height': height,
        'steps': num_inference_steps,
        'cfg': guidance_scale,
        'denoise': denoising_strength,
        'seed': seed,
        'gen_type': gen_type
    }
    canny_request_body = {
        'controlnet_requests':
            [
                {
                    'controlnet': canny_model_name,
                    'type': 'canny',
                    'image': canny_image,
                    'strength': canny_control_weight,
                    'start_percent': canny_start,
                    'end_percent': canny_end,
                }
            ]
    }
    if canny_enable :
        request_body.update(canny_request_body)

    url = f"http://{ip_addr}/sd3/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image

if __name__ == "__main__":
    ip_addr = '117.52.72.83'

    request_body = {
        'checkpoint': 'SD3_sd3_medium_incl_clips_t5xxlfp16.safetensors',
        "prompt_positive": 'a dog',
        "prompt_negative": "",
        'width': 1344,
        'height': 768,
        'steps': 20,
        'cfg': 4,
        'denoise': 1,
        'gen_type': 't2i'
    }

    url = f"http://{ip_addr}:7863/sd3/generate"
    response = requests.post(url, json=request_body)