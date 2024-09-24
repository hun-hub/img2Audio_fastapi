from fastapi import APIRouter, HTTPException
import torch
from utils import set_comfyui_packages
from utils.loader import load_checkpoint
from utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from .utils import (model_sampling_flux,
                    get_init_noise,
                    flux_guidance,
                    construct_condition)
from utils.comfyui import (encode_prompt,
                           sample_image,
                           decode_latent,
                           encode_image,
                           encode_image_for_inpaint,
                           mask_blur,
                           apply_lora_to_unet)
import random

# set_comfyui_packages()
router = APIRouter()

@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['flux'][1]
    vae = cached_model_dict['vae']['flux'][1]
    clip = cached_model_dict['clip']['flux'][1]

    start_base = int(request_data.steps - request_data.steps * request_data.denoise)
    end_base = request_data.steps

    if request_data.gen_type == 't2i' :
        init_noise = get_init_noise(request_data.width,
                                     request_data.height,
                                     request_data.batch_size)
    elif request_data.gen_type == 'i2i' and request_data.init_image:
        init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
        init_noise = encode_image(vae, init_image)
    elif request_data.gen_type == 'inpaint' and request_data.init_image and request_data.mask:
        init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
        mask = convert_base64_to_image_tensor(request_data.mask)[:, :, :, 0] / 255
        mask = mask_blur(mask)
        init_noise = encode_image_for_inpaint(vae, init_image, mask, grow_mask_by=0)
    else :
        raise ValueError("Invalid generation type and image: {}".format(request_data.gen_type))

    b, c, h, w = init_noise['samples'].size()

    lora_requests = request_data.lora_requests
    canny_request = None
    depth_request = None
    for controlnet_request in request_data.controlnet_requests:
        if controlnet_request.type == 'canny':
            canny_request = controlnet_request
        if controlnet_request.type == 'depth':
            depth_request = controlnet_request

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed
    if lora_requests :
        for lora_request in lora_requests :
            unet, _ = apply_lora_to_unet(
                unet,
                None,
                cached_model_dict,
                lora_request)

    positive_cond, negative_cond = encode_prompt(clip,
                                                 request_data.prompt_positive,
                                                 request_data.prompt_negative)

    positive_cond = flux_guidance(positive_cond, request_data.cfg)

    unet, positive_cond, negative_cond = construct_condition(
        unet,
        cached_model_dict,
        positive_cond,
        negative_cond,
        canny_request,
        depth_request,
    )

    unet = model_sampling_flux(unet, width = int(w * 8), height = int(h * 8))

    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond,
        negative_cond= negative_cond,
        latent_image= init_noise,
        seed= seed,
        steps= request_data.steps,
        cfg= 1.0,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step = start_base,
        end_at_step = end_base,
    )

    image_tensor = decode_latent(vae, latent_image)
    if request_data.gen_type == 'inpaint':
        image_tensor = image_tensor * mask.unsqueeze(-1) + init_image * (1 - mask.unsqueeze(-1))

    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    return image_base64

