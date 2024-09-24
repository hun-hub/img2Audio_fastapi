import torch
from utils import set_comfyui_packages
from utils.loader import get_function_from_comfyui
import requests
from utils.handler import handle_response
from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 convert_base64_to_image_tensor,
                                 controlnet_image_preprocess
                                 )
from utils.comfyui import (apply_controlnet,
                           apply_ipadapter,
                           make_image_batch)

from types import NoneType
from PIL import Image

def flux_guidance(text_embed, cfg) :
    from ComfyUI.comfy_extras.nodes_flux import FluxGuidance
    flux_guidance = FluxGuidance()
    text_embed = flux_guidance.append(text_embed, cfg)[0]
    return text_embed

def model_sampling_flux(unet, max_shift=1.15, base_shift=0.5, width=1024, height=1024) :
    from ComfyUI.comfy_extras.nodes_model_advanced import ModelSamplingFlux
    model_sampler = ModelSamplingFlux()
    unet = model_sampler.patch(unet, max_shift, base_shift, width, height)[0]

    return unet

def get_init_noise(width, height, batch_size=1) :
    from ComfyUI.comfy_extras.nodes_sd3 import EmptySD3LatentImage
    latent_sampler = EmptySD3LatentImage()
    init_noise = latent_sampler.generate(width, height, batch_size)[0]

    return init_noise

def construct_condition(unet,
                        cached_model_dict,
                        positive,
                        negative,
                        canny_request,
                        depth_request,
                        ):

    if canny_request is not None :
        control_image = convert_base64_to_image_tensor(canny_request.image) / 255
        control_image = controlnet_image_preprocess(control_image, 'canny', 'sdxl', )
        controlnet = cached_model_dict['controlnet']['flux'][canny_request.type][1]
        positive, negative = apply_controlnet(
            positive,
            negative,
            controlnet,
            control_image,
            canny_request.strength,
            canny_request.start_percent,
            canny_request.end_percent,)
    # if depth_request is not None :
    #     control_image = convert_base64_to_image_tensor(depth_request.image) / 255
    #     control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
    #     control_image = torch.where(control_mask[:, :, :, None] > 0.5, 1, control_image)
    #     controlnet = cached_model_dict['controlnet']['flux'][depth_request.type][1]
    #     positive, negative = apply_controlnet(positive,
    #                                           negative,
    #                                           controlnet,
    #                                           control_image,
    #                                           depth_request.strength,
    #                                           depth_request.start_percent,
    #                                           depth_request.end_percent, )

    return unet, positive, negative

def sned_flux_request_to_api(
        unet_name,
        vae_name,
        clip_1,
        clip_2,
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

        lora_enable,
        lora_model_name_1,
        strength_model_1,
        strength_clip_1,
        lora_model_name_2,
        strength_model_2,
        strength_clip_2,
        lora_model_name_3,
        strength_model_3,
        strength_clip_3,

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
        'unet': unet_name,
        'vae': vae_name,
        'clip': (clip_1, clip_2),

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
        'lora_requests': [],
    }

    canny_body = {
        'controlnet': canny_model_name,
        'type': 'canny',
        'image': canny_image,
        'strength': canny_control_weight,
        'start_percent': canny_start,
        'end_percent': canny_end,
    }

    lora_requests_sorted = sorted([[lora_model_name_1, strength_model_1, strength_clip_1],
                                   [lora_model_name_2, strength_model_2, strength_clip_2],
                                   [lora_model_name_3, strength_model_3, strength_clip_3]])
    lora_body_list = []

    for lora_request_sorted in lora_requests_sorted:
        if lora_request_sorted[0] == 'None': continue
        lora_body = {'lora': lora_request_sorted[0],
                     'strength_model': lora_request_sorted[1],
                     'strength_clip': lora_request_sorted[2], }
        lora_body_list.append(lora_body)

    if canny_enable :
        request_body['controlnet_requests'].append(canny_body)
    if lora_enable :
        for lora_body in lora_body_list:
            if lora_body['lora'] != 'None' :
                request_body['lora_requests'].append(lora_body)

    url = f"http://{ip_addr}/flux/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image