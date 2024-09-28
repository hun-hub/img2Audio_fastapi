import torch
import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess
                                      )
from cgen_utils.comfyui import (apply_controlnet,
                                apply_ipadapter,
                                make_image_batch)

from types import NoneType
from PIL import Image
@torch.inference_mode()
def flux_guidance(text_embed, cfg) :
    from ComfyUI.comfy_extras.nodes_flux import FluxGuidance
    flux_guidance = FluxGuidance()
    text_embed = flux_guidance.append(text_embed, cfg)[0]
    return text_embed
@torch.inference_mode()
def model_sampling_flux(unet, max_shift=1.15, base_shift=0.5, width=1024, height=1024) :
    from ComfyUI.comfy_extras.nodes_model_advanced import ModelSamplingFlux
    model_sampler = ModelSamplingFlux()
    unet = model_sampler.patch(unet, max_shift, base_shift, width, height)[0]

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
        controlnet = cached_model_dict['controlnet']['flux'][controlnet_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              controlnet_request.strength,
                                              controlnet_request.start_percent,
                                              controlnet_request.end_percent, )


    return positive, negative



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
        canny_preprocessor_type,
        canny_control_weight,
        canny_start,
        canny_end,

        depth_enable,
        depth_model_name,
        depth_image,
        depth_preprocessor_type,
        depth_control_weight,
        depth_start,
        depth_end,

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
    if not isinstance(depth_image, NoneType):
        depth_image = resize_image_for_sd(Image.fromarray(depth_image))
        depth_image = convert_image_to_base64(depth_image)

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
        'preprocessor_type': canny_preprocessor_type,
        'strength': canny_control_weight,
        'start_percent': canny_start,
        'end_percent': canny_end,
    }

    depth_body = {
        'controlnet': depth_model_name,
        'type': 'depth',
        'image': depth_image,
        'preprocessor_type': depth_preprocessor_type,
        'strength': depth_control_weight,
        'start_percent': depth_start,
        'end_percent': depth_end,
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
    if depth_enable :
        request_body['controlnet_requests'].append(depth_body)

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