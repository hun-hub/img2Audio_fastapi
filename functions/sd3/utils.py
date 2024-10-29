import torch
import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      )
from types import NoneType
from PIL import Image
import httpx
import asyncio



@torch.inference_mode()
def apply_controlnet(positive, negative, controlnet, vae, image, strength, start_percent, end_percent,) :
    from ComfyUI.comfy_extras.nodes_sd3 import ControlNetApplySD3
    controlnet_applier = ControlNetApplySD3()
    positive, negative = controlnet_applier.apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent, vae)
    return positive, negative
@torch.inference_mode()
def model_sampling_sd3(unet, shift:float = 3) :
    from ComfyUI.comfy_extras.nodes_model_advanced import ModelSamplingSD3
    model_sampler = ModelSamplingSD3()
    unet = model_sampler.patch(unet, shift)[0]

    return unet
@torch.inference_mode()
def get_init_noise(width, height, batch_size=1) :
    from ComfyUI.comfy_extras.nodes_sd3 import EmptySD3LatentImage
    latent_sampler = EmptySD3LatentImage()
    init_noise = latent_sampler.generate(width, height, batch_size)[0]

    return init_noise


@torch.inference_mode()
def construct_controlnet_condition(
        cached_model_dict,
        positive,
        negative,
        controlnet_requests,
):

    for controlnet_request in controlnet_requests:
        if controlnet_request.type == 'inpaint':
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
            control_image = torch.where(control_mask[:, :, :, None] > 0.5, 1, control_image)
        else :
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image = controlnet_image_preprocess(control_image, controlnet_request.preprocessor_type, 'sdxl')
        controlnet = cached_model_dict['controlnet']['sd3'][controlnet_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              controlnet_request.strength,
                                              controlnet_request.start_percent,
                                              controlnet_request.end_percent, )


    return positive, negative





async def sned_sd3_request_to_api(
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
        canny_preprocessor_type,
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
        'gen_type': gen_type,
        'controlnet_requests': [],
    }

    canny_body = {
        'controlnet': canny_model_name,
        'type': 'canny',
        'image': canny_image,
        'preprocessor_type': canny_preprocessor_type,
        'strength': canny_control_weight,
        'start_percent': canny_start,
        'end_percent': canny_end,
    }

    if canny_enable :
        request_body['controlnet_requests'].append(canny_body)

    url = f"http://{ip_addr}/sd3/generate"
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json=request_body)
        data = handle_response(response)
        image_base64 = data['image_base64']
        image = convert_base64_to_image_array(image_base64)
        return [image]

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