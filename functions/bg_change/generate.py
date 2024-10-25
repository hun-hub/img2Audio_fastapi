import torch
from .utils import construct_ipadapter_condition, construct_controlnet_condition, encode_prompt_advance
from cgen_utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from cgen_utils.comfyui import (encode_prompt,
                                sample_image,
                                decode_latent,
                                encode_image,
                                encode_image_for_inpaint,
                                apply_controlnet,
                                get_init_noise,
                                apply_lora_to_unet,
                                mask_blur)
from ComfyUI.comfy_extras.nodes_pag import PerturbedAttentionGuidance
import random

@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['sd15']['base'][1]
    vae = cached_model_dict['vae']['sd15']['base'][1]
    clip = cached_model_dict['clip']['sd15']['base'][1]

    init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
    mask = convert_base64_to_image_tensor(request_data.mask)[:, :, :, 0] / 255

    control_image = torch.where(mask[:, :, :, None] < 0.5, init_image, 0)
    control_mask = 1 - mask

    b, h, w, c = init_image.size()
    init_noise = get_init_noise(w, h, 1)

    ipadapter_request = request_data.ipadapter_request
    lora_requests = request_data.lora_requests
    controlnet_requests = request_data.controlnet_requests

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed
    for lora_request in lora_requests :
        unet, clip = apply_lora_to_unet(
            unet,
            clip,
            cached_model_dict,
            lora_request)

    unet = construct_ipadapter_condition(
        unet,
        cached_model_dict,
        ipadapter_request
    )

    positive_cond, negative_cond = encode_prompt_advance(
        clip,
        request_data.is_retouch,
        request_data.prompt_positive,
        request_data.prompt_negative,
        mask,
        request_data.prompt_retouch)


    positive_cond_first, negative_cond_first = construct_controlnet_condition(
        cached_model_dict,
        positive_cond,
        negative_cond,
        control_image,
        control_mask,
        request_data.is_retouch,
        True,
        controlnet_requests,
    )

    positive_cond_second, negative_cond_second = construct_controlnet_condition(
        cached_model_dict,
        positive_cond,
        negative_cond,
        control_image,
        control_mask,
        request_data.is_retouch,
        False,
        controlnet_requests,
    )

    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond_first,
        negative_cond= negative_cond_first,
        latent_image= init_noise,
        seed= seed,
        steps= 25,
        cfg= 7,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step= 0,
        end_at_step= 1000,)

    unet = PerturbedAttentionGuidance().patch(unet, 3)[0]
    latent_image = sample_image(
        unet= unet,
        positive_cond= positive_cond_second,
        negative_cond= negative_cond_second,
        latent_image= latent_image,
        seed= seed,
        steps= 25,
        cfg= 7,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        start_at_step= 17,
        end_at_step= 1000,)

    image_tensor = decode_latent(vae, latent_image)
    image_tensor_blend = image_tensor * mask.unsqueeze(-1) + init_image * (1 - mask.unsqueeze(-1))

    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    image_blend_base64 = convert_image_tensor_to_base64(image_tensor_blend * 255)

    return image_base64, image_blend_base64

