import torch
from cgen_utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from cgen_utils.comfyui import (encode_prompt,
                                sample_image,
                                decode_latent,
                                encode_image,
                                encode_image_for_inpaint,
                                apply_lora_to_unet,
                                get_init_noise,
                                mask_blur)
from .utils import construct_controlnet_condition, construct_ipadapter_condition

import random
import copy


@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['sdxl']['base'][1].clone()
    vae = cached_model_dict['vae']['sdxl']['base'][1]
    clip = cached_model_dict['clip']['sdxl']['base'][1]

    start = 0
    middle = int(request_data.steps - request_data.steps * request_data.denoise)
    end = request_data.steps

    init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
    mask = convert_base64_to_image_tensor(request_data.mask)[:, :, :, 0] / 255
    mask = mask_blur(mask)
    init_noise = encode_image_for_inpaint(vae, init_image, mask, grow_mask_by=0)

    ipadapter_request = request_data.ipadapter_request
    controlnet_requests = request_data.controlnet_requests

    unet = construct_ipadapter_condition(
        unet,
        cached_model_dict,
        ipadapter_request
    )

    # Base Model Flow
    positive_cond, negative_cond = encode_prompt(clip,
                                                 request_data.prompt_positive,
                                                 request_data.prompt_negative)

    positive_cond_controlnet, negative_cond_controlnet = construct_controlnet_condition(
        cached_model_dict,
        positive_cond,
        negative_cond,
        controlnet_requests,
    )


    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed
    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond_controlnet,
        negative_cond= negative_cond_controlnet,
        latent_image= init_noise,
        seed= seed,
        steps= request_data.steps,
        cfg= request_data.cfg,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step= start,
        end_at_step= middle,
        add_noise = 'enable',
        return_with_leftover_noise='enable'
    )

    del latent_image['noise_mask']

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed
    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond,
        negative_cond= negative_cond,
        latent_image= latent_image,
        seed= seed,
        steps= request_data.steps,
        cfg= request_data.cfg,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step= middle,
        end_at_step= end,
        add_noise='disable',
        return_with_leftover_noise='disable'
    )

    image_tensor = decode_latent(vae, latent_image)
    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    return image_base64
