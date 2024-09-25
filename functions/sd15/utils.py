import torch
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


def construct_condition(unet,
                        cached_model_dict,
                        positive,
                        negative,
                        controlnet_requests,
                        ipadapter_request,
                        ):
    for controlnet_request in controlnet_requests:
        if controlnet_request.type == 'inpaint':
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image, control_mask = control_image[:, :, :, :3], control_image[:, :, :, 3]
            control_image = torch.where(control_mask[:, :, :, None] > 0.5, 1, control_image)
        else:
            control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
            control_image = controlnet_image_preprocess(control_image, controlnet_request.preprocessor_type, 'sd15')
        controlnet = cached_model_dict['controlnet']['sd15'][controlnet_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              controlnet_request.strength,
                                              controlnet_request.start_percent,
                                              controlnet_request.end_percent, )
    if ipadapter_request is not None:
        clip_vision = load_clip_vision(ipadapter_request.clip_vision)
        ipadapter = cached_model_dict['ipadapter']['sd15'][1]
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

def sned_sd15_request_to_api(
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

        inpaint_enable,
        inpaint_model_name,
        inpaint_image,
        inpaint_mask,
        inpaint_preprocessor_type,
        inpaint_control_weight,
        inpaint_start,
        inpaint_end,

        depth_enable,
        depth_model_name,
        depth_image,
        depth_preprocessor_type,
        depth_control_weight,
        depth_start,
        depth_end,

        normal_enable,
        normal_model_name,
        normal_image,
        normal_preprocessor_type,
        normal_control_weight,
        normal_start,
        normal_end,

        pose_enable,
        pose_model_name,
        pose_image,
        pose_preprocessor_type,
        pose_control_weight,
        pose_start,
        pose_end,

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
    sd_resolution = 1024
    height, width = eval(resolution)
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image), resolution=sd_resolution)
        image = convert_image_to_base64(image)
    if not isinstance(mask, NoneType):
        mask = resize_image_for_sd(Image.fromarray(mask), is_mask=True, resolution=sd_resolution)
        mask = convert_image_to_base64(mask)
    if not isinstance(canny_image, NoneType):
        canny_image = resize_image_for_sd(Image.fromarray(canny_image), resolution=sd_resolution)
        canny_image = convert_image_to_base64(canny_image)
    if not isinstance(inpaint_image, NoneType) and not isinstance(inpaint_mask, NoneType):
        inpaint_image = resize_image_for_sd(Image.fromarray(inpaint_image), resolution=sd_resolution)
        inpaint_mask = resize_image_for_sd(Image.fromarray(inpaint_mask[:, :, 0]), is_mask=True, resolution=sd_resolution)

        inpaint_image_arr = np.array(inpaint_image)
        inpaint_mask_arr = np.expand_dims(np.array(inpaint_mask), axis=2)
        inpaint_image_arr = np.concatenate((inpaint_image_arr, inpaint_mask_arr), axis=2)
        inpaint_image = Image.fromarray(inpaint_image_arr)

        inpaint_image = convert_image_to_base64(inpaint_image)
    if not isinstance(depth_image, NoneType):
        depth_image = resize_image_for_sd(Image.fromarray(depth_image), resolution=sd_resolution)
        depth_image = convert_image_to_base64(depth_image)
    if not isinstance(normal_image, NoneType):
        normal_image = resize_image_for_sd(Image.fromarray(normal_image), resolution=sd_resolution)
        normal_image = convert_image_to_base64(normal_image)
    if not isinstance(pose_image, NoneType):
        pose_image = resize_image_for_sd(Image.fromarray(pose_image), resolution=sd_resolution)
        pose_image = convert_image_to_base64(pose_image)
    if not isinstance(ipadapter_images, NoneType):
        ipadapter_images = [resize_image_for_sd(Image.fromarray(ipadapter_image[0]), resolution=sd_resolution) for ipadapter_image in ipadapter_images]
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

    canny_body = {
        'controlnet': canny_model_name,
        'type': 'canny',
        'image': canny_image,
        'preprocessor_type': canny_preprocessor_type,
        'strength': canny_control_weight,
        'start_percent': canny_start,
        'end_percent': canny_end,
    }

    inpaint_body = {
        'controlnet': inpaint_model_name,
        'type': 'inpaint',
        'image': inpaint_image,
        'preprocessor_type': inpaint_preprocessor_type,
        'strength': inpaint_control_weight,
        'start_percent': inpaint_start,
        'end_percent': inpaint_end,
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

    normal_body = {
        'controlnet': normal_model_name,
        'type': 'normal',
        'image': normal_image,
        'preprocessor_type': normal_preprocessor_type,
        'strength': normal_control_weight,
        'start_percent': normal_start,
        'end_percent': normal_end,
    }

    pose_body = {
        'controlnet': pose_model_name,
        'type': 'pose',
        'image': pose_image,
        'preprocessor_type': pose_preprocessor_type,
        'strength': pose_control_weight,
        'start_percent': pose_start,
        'end_percent': pose_end,
    }

    ipadapter_body = {
        'ipadapter': ipadapter_model_name,
        'clip_vision': 'CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors',
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
        if lora_request_sorted[0] == 'None': continue
        lora_body = {'lora': lora_request_sorted[0],
                     'strength_model': lora_request_sorted[1],
                     'strength_clip': lora_request_sorted[2], }
        lora_body_list.append(lora_body)

    if canny_enable :
        request_body['controlnet_requests'].append(canny_body)
    if inpaint_enable :
        request_body['controlnet_requests'].append(inpaint_body)
    if depth_enable :
        request_body['controlnet_requests'].append(depth_body)
    if normal_enable :
        request_body['controlnet_requests'].append(normal_body)
    if pose_enable :
        request_body['controlnet_requests'].append(pose_body)
    if ipadapter_enable :
        request_body['ipadapter_request'] = ipadapter_body
    # TODO: 그냥 extend 한줄로 해도 될듯. 위에서 None 걸러내서.
    if lora_enable :
        for lora_body in lora_body_list:
            if lora_body['lora'] != 'None' :
                request_body['lora_requests'].append(lora_body)


    url = f"http://{ip_addr}/sd15/generate"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image