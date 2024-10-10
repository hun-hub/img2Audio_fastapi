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

from cgen_utils.loader import load_clip_vision
from types import NoneType
from PIL import Image
import numpy as np

@torch.inference_mode()
def construct_controlnet_condition(
        cached_model_dict,
        positive,
        negative,
        controlnet_requests,
):

    for controlnet_request in controlnet_requests:
        control_image = convert_base64_to_image_tensor(controlnet_request.image) / 255
        control_image = controlnet_image_preprocess(control_image, controlnet_request.preprocessor_type, 'sdxl')
        controlnet = cached_model_dict['controlnet']['sdxl'][controlnet_request.type][1]
        positive, negative = apply_controlnet(positive,
                                              negative,
                                              controlnet,
                                              control_image,
                                              controlnet_request.strength,
                                              controlnet_request.start_percent,
                                              controlnet_request.end_percent, )


    return positive, negative


def sned_half_inpainting_request_to_api(
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
    if not isinstance(normal_image, NoneType):
        normal_image = resize_image_for_sd(Image.fromarray(normal_image))
        normal_image = convert_image_to_base64(normal_image)
    if not isinstance(pose_image, NoneType):
        pose_image = resize_image_for_sd(Image.fromarray(pose_image))
        pose_image = convert_image_to_base64(pose_image)

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
        'controlnet_requests': []
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

    if canny_enable :
        request_body['controlnet_requests'].append(canny_body)
    if depth_enable :
        request_body['controlnet_requests'].append(depth_body)
    if normal_enable :
        request_body['controlnet_requests'].append(normal_body)
    if pose_enable :
        request_body['controlnet_requests'].append(pose_body)

    url = f"http://{ip_addr}/sdxl/half_inpainting"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return [image]