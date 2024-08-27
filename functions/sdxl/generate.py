import torch
from .utils import construct_condition
from utils import set_comfyui_packages
from utils.loader import load_checkpoint, apply_lora_to_unet
from utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from utils.comfyui import (encode_prompt,
                           sample_image,
                           decode_latent,
                           encode_image,
                           encode_image_for_inpaint,
                           apply_controlnet,
                           make_canny,
                           get_init_noise,
                           mask_blur)
import random

# set_comfyui_packages()

@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet_base = cached_model_dict['unet']['sdxl']['base'][1]
    vae_base = cached_model_dict['vae']['sdxl']['base'][1]
    clip_base = cached_model_dict['clip']['sdxl']['base'][1]
    vae_refine = vae_base
    start_base = 0
    end_base = request_data.steps

    if request_data.refiner is not None :
        end_base = int(request_data.steps * request_data.refine_switch)

    if request_data.gen_type == 't2i' :
        init_noise = get_init_noise(request_data.width,
                                     request_data.height,
                                     request_data.batch_size)
    elif request_data.gen_type == 'i2i' and request_data.init_image:
        init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
        init_noise = encode_image(vae_base, init_image)
    elif request_data.gen_type == 'inpaint' and request_data.init_image and request_data.mask:
        init_image = convert_base64_to_image_tensor(request_data.init_image) / 255
        mask = convert_base64_to_image_tensor(request_data.mask)[:, :, :, 0] / 255
        mask = mask_blur(mask)
        init_noise = encode_image_for_inpaint(vae_base, init_image, mask, grow_mask_by=0)
    else :
        raise ValueError("Invalid generation type and image: {}".format(request_data.gen_type))

    ipadapter_request = request_data.ipadapter_request
    lora_requests = request_data.lora_requests
    canny_request = None
    inpaint_request = None
    for controlnet_request in request_data.controlnet_requests :
        if controlnet_request.type == 'canny' :
            canny_request = controlnet_request
        if controlnet_request.type == 'inpaint' :
            inpaint_request = controlnet_request

    seed = random.randint(1, int(1e9)) if request_data.seed == -1 else request_data.seed
    if lora_requests :
        for lora_request in lora_requests :
            unet_base, clip_base = apply_lora_to_unet(unet_base, clip_base, lora_request.lora, lora_request.strength_model, lora_request.strength_clip)
    # Base Model Flow
    positive_cond, negative_cond = encode_prompt(clip_base,
                                                 request_data.prompt_positive,
                                                 request_data.prompt_negative)

    unet_base, positive_cond_base, negative_cond_base = construct_condition(
        unet_base,
        cached_model_dict,
        positive_cond,
        negative_cond,
        canny_request,
        inpaint_request,
        ipadapter_request,
    )

    latent_image = sample_image(
        unet= unet_base,
        positive_cond= positive_cond_base,
        negative_cond= negative_cond_base,
        latent_image= init_noise,
        seed= seed,
        steps= request_data.steps,
        cfg= request_data.cfg,
        sampler_name= request_data.sampler_name,
        scheduler= request_data.scheduler,
        denoise= request_data.denoise,
        start_at_step= start_base,
        end_at_step= end_base,)

    if request_data.refiner is not None :
        unet_refine = cached_model_dict['unet']['sdxl']['refiner'][1]
        vae_refine = cached_model_dict['vae']['sdxl']['refiner'][1]
        clip_refine = cached_model_dict['clip']['sdxl']['refiner'][1]

        start_refine = end_base
        end_refine = request_data.steps

        if lora_requests:
            for lora_request in lora_requests:
                unet_refine, clip_refine = apply_lora_to_unet(
                    unet_refine,
                    clip_refine,
                    lora_request.lora,
                    lora_request.strength_model,
                    lora_request.strength_clip)

        positive_cond, negative_cond = encode_prompt(clip_refine,
                                                     request_data.prompt_positive,
                                                     request_data.prompt_negative)
        unet_refine, positive_cond_refine, negative_cond_refine = construct_condition(unet_refine,
                                                                                      cached_model_dict,
                                                                                      positive_cond,
                                                                                      negative_cond,
                                                                                      canny_request,
                                                                                      inpaint_request,
                                                                                      ipadapter_request)
        latent_image = sample_image(
            unet=unet_refine,
            positive_cond=positive_cond,
            negative_cond=negative_cond,
            latent_image=latent_image,
            seed=seed,
            steps=request_data.steps,
            cfg=request_data.cfg,
            sampler_name=request_data.sampler_name,
            scheduler=request_data.scheduler,
            denoise=request_data.denoise,
            start_at_step=start_refine,
            end_at_step=end_refine, )



    image_tensor = decode_latent(vae_refine, latent_image)
    if request_data.gen_type == 'inpaint' :
        image_tensor = image_tensor * mask.unsqueeze(-1) + init_image * (1 - mask.unsqueeze(-1))

    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    return image_base64
