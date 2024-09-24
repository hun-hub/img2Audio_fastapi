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

from utils.loader import load_clip_vision
from types import NoneType
from PIL import Image
import numpy as np

# set_comfyui_packages()
def construct_condition(unet,
                        cached_model_dict,
                        positive,
                        negative,
                        canny_request,
                        inpaint_request,
                        ipadapter_request,
                        ):

    if canny_request is not None :
        control_image = convert_base64_to_image_tensor(canny_request.image) / 255
        control_image = controlnet_image_preprocess(control_image, 'canny', 'sdxl')
        controlnet = cached_model_dict['controlnet']['sdxl'][canny_request.type][1]
        positive, negative = apply_controlnet(positive,
                                                        negative,
                                                        controlnet,
                                                        control_image,
                                                        canny_request.strength,
                                                        canny_request.start_percent,
                                                        canny_request.end_percent,)
    if inpaint_request is not None :
        control_image = convert_base64_to_image_tensor(inpaint_request.image) / 255
        control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
        control_image = torch.where(control_mask[:, :, :, None] > 0.5, 1, control_image)
        controlnet = cached_model_dict['controlnet']['sdxl'][inpaint_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              inpaint_request.strength,
                                              inpaint_request.start_percent,
                                              inpaint_request.end_percent, )
    if ipadapter_request is not None:
        clip_vision = load_clip_vision(ipadapter_request.clip_vision)
        ipadapter = cached_model_dict['ipadapter']['sdxl'][1]
        ipadapter_images = [convert_base64_to_image_tensor(image) / 255 for image in ipadapter_request.images]
        image_batch = make_image_batch(ipadapter_images)
        unet = apply_ipadapter(model= unet,
                               ipadapter=ipadapter,
                               clip_vision=clip_vision,
                               image= image_batch,
                               weight= ipadapter_request.weight,
                               start_at = ipadapter_request.start_at,
                               end_at = ipadapter_request.end_at,
                               weight_type = ipadapter_request.weight_type,
                               combine_embeds = ipadapter_request.combine_embeds,
                               embeds_scaling= ipadapter_request.embeds_scaling,)
    return unet, positive, negative

def sned_sdxl_request_to_api(
        checkpoint,
        image,
        mask,
        prompt,
        resolution,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,

        refiner_enable,
        refiner_name,
        refine_switch,

        canny_enable,
        canny_model_name,
        canny_image,
        canny_control_weight,
        canny_start,
        canny_end,

        inpaint_enable,
        inpaint_model_name,
        inpaint_image,
        inpaint_mask,
        inpaint_control_weight,
        inpaint_start,
        inpaint_end,

        ipadapter_enable,
        ipadapter_model_name,
        ipadapter_images,
        ipadapter_weight,
        ipadapter_start,
        ipadapter_end,

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
    if not isinstance(inpaint_image, NoneType) and not isinstance(inpaint_mask, NoneType):
        inpaint_image = resize_image_for_sd(Image.fromarray(inpaint_image))
        inpaint_mask = resize_image_for_sd(Image.fromarray(inpaint_mask[:, :, 0]), is_mask=True)

        inpaint_image_arr = np.array(inpaint_image)
        inpaint_mask_arr = np.expand_dims(np.array(inpaint_mask), axis=2)
        inpaint_image_arr = np.concatenate((inpaint_image_arr, inpaint_mask_arr), axis=2)
        inpaint_image = Image.fromarray(inpaint_image_arr)

        inpaint_image = convert_image_to_base64(inpaint_image)
    if not isinstance(ipadapter_images, NoneType):
        ipadapter_images = [resize_image_for_sd(Image.fromarray(ipadapter_image[0])) for ipadapter_image in ipadapter_images]
        ipadapter_images = [convert_image_to_base64(ipadapter_image) for ipadapter_image in ipadapter_images]

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
        'lora_requests': [],
    }

    refiner_body = {
        'refiner': refiner_name,
        'refine_switch': refine_switch,
    }

    canny_body = {
        'controlnet': canny_model_name,
        'type': 'canny',
        'image': canny_image,
        'strength': canny_control_weight,
        'start_percent': canny_start,
        'end_percent': canny_end,
    }

    inpaint_body = {
        'controlnet': inpaint_model_name,
        'type': 'inpaint',
        'image': inpaint_image,
        'strength': inpaint_control_weight,
        'start_percent': inpaint_start,
        'end_percent': inpaint_end,
    }

    ipadapter_body = {
        'ipadapter': ipadapter_model_name,
        'clip_vision': 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors',
        'images': ipadapter_images,
        'weight': ipadapter_weight,
        'start_at': ipadapter_start,
        'end_at': ipadapter_end,
    }

    lora_requests_sorted = sorted([[lora_model_name_1, strength_model_1, strength_clip_1],
                                   [lora_model_name_2, strength_model_2, strength_clip_2],
                                   [lora_model_name_3, strength_model_3, strength_clip_3]])
    lora_body_list = []

    for lora_request_sorted in lora_requests_sorted:
        if lora_request_sorted[0] == 'None' : continue
        lora_body = {'lora': lora_request_sorted[0],
                     'strength_model': lora_request_sorted[1],
                     'strength_clip': lora_request_sorted[2],}
        lora_body_list.append(lora_body)


    if refiner_enable:
        request_body.update(refiner_body)
    if canny_enable :
        request_body['controlnet_requests'].append(canny_body)
    if inpaint_enable :
        request_body['controlnet_requests'].append(inpaint_body)
    if ipadapter_enable :
        request_body['ipadapter_request'] = ipadapter_body
    if lora_enable :
        for lora_body in lora_body_list:
            if lora_body['lora'] != 'None' :
                request_body['lora_requests'].append(lora_body)

    url = f"http://{ip_addr}/sdxl/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image