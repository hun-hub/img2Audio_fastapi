import torch
from .utils import construct_condition
from utils import set_comfyui_packages
from utils.loader import load_checkpoint
from utils.image_process import convert_image_tensor_to_base64, convert_base64_to_image_tensor
from utils.comfyui import (encode_prompt,
                           sample_image,
                           decode_latent,
                           encode_image,
                           encode_image_for_inpaint,
                           apply_controlnet,
                           make_canny,
                           get_init_noise,
                           apply_lora_to_unet,
                           mask_blur)
import random

# set_comfyui_packages()

# prompt_post_fix = ", RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
scale_factor = 0.18215
@torch.inference_mode()
def generate_image(cached_model_dict, request_data):
    unet = cached_model_dict['unet']['sd15'][1]
    vae = cached_model_dict['vae']['sd15'][1]
    clip = cached_model_dict['clip']['sd15'][1]
    start_base = 0
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
            unet, clip = apply_lora_to_unet(
                unet,
                clip,
                cached_model_dict,
                lora_request)

    positive_cond, negative_cond = encode_prompt(clip,
                                                 request_data.prompt_positive ,
                                                 request_data.prompt_negative)

    unet, positive_cond, negative_cond = construct_condition(
        unet,
        cached_model_dict,
        positive_cond,
        negative_cond,
        canny_request,
        inpaint_request,
        ipadapter_request)

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
        denoise= request_data.denoise,
        start_at_step= start_base,
        end_at_step= end_base,)

    image_tensor = decode_latent(vae, latent_image)
    if request_data.gen_type == 'inpaint' :
        image_tensor = image_tensor * mask.unsqueeze(-1) + init_image * (1 - mask.unsqueeze(-1))

    if inpaint_request is not None:
        control_image = convert_base64_to_image_tensor(inpaint_request.image) / 255
        control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
        if control_image.squeeze().size() == image_tensor.squeeze().size():
            control_mask = mask_blur(control_mask)
            image_tensor = image_tensor * control_mask.unsqueeze(-1) + control_image * (1 - control_mask.unsqueeze(-1))

    image_base64 = convert_image_tensor_to_base64(image_tensor * 255)
    return image_base64

