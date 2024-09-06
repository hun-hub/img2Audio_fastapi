from fastapi import APIRouter, HTTPException
import torch
from utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from .utils import model_sampling_sd3, get_init_noise, apply_controlnet
from utils.comfyui import (encode_prompt,
                           sample_image,
                           decode_latent,
                           encode_image,
                           encode_image_for_inpaint,
                           make_canny,
                           mask_blur)
import random

# set_comfyui_packages()
router = APIRouter()

@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['sd3'][1]
    vae = cached_model_dict['vae']['sd3'][1]
    clip = cached_model_dict['clip']['sd3'][1]

    unet = model_sampling_sd3(unet)

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

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed

    positive_cond, negative_cond = encode_prompt(clip,
                                                 request_data.prompt_positive,
                                                 request_data.prompt_negative)

    for controlnet_request in request_data.controlnet_requests :
        control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
        control_image = make_canny(control_image)
        controlnet = cached_model_dict['controlnet']['sd3'][controlnet_request.type][1]
        positive_cond, negative_cond = apply_controlnet(positive_cond,
                                                        negative_cond,
                                                        controlnet,
                                                        vae,
                                                        control_image,
                                                        controlnet_request.strength,
                                                        controlnet_request.start_percent,
                                                        controlnet_request.end_percent,)

    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond,
        negative_cond= negative_cond,
        latent_image= init_noise,
        seed= seed,
        steps= request_data.steps,
        cfg= request_data.cfg,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        denoise= request_data.denoise,)

    image_tensor = decode_latent(vae, latent_image)
    if request_data.gen_type == 'inpaint':
        image_tensor = image_tensor * mask.unsqueeze(-1) + init_image * (1 - mask.unsqueeze(-1))

    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    return image_base64

