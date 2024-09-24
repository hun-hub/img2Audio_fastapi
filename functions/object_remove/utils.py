import torch
from utils import set_comfyui_packages
import requests
from utils.handler import handle_response
from utils.image_process import (convert_base64_to_image_array,
                                 convert_image_array_to_base64,
                                 resize_image_for_sd,
                                 convert_image_to_base64,
                                 convert_base64_to_image_tensor
                                 )
from utils.comfyui import (apply_controlnet,
                           apply_ipadapter,
                           make_image_batch,
                           encode_image_for_inpaint)

from utils.loader import load_clip_vision, resize_image_with_pad, load_lamaInpainting, load_fooocus, get_function_from_comfyui
from types import NoneType
from PIL import Image
import numpy as np
from urllib.parse import urlparse
import os
import cv2

# set_comfyui_packages()

def construct_condition(unet,
                        vae,
                        image,
                        mask,
                        inpaint_model_name
                        ):

    inpaint_patch = load_fooocus(inpaint_model_name)
    latent = encode_image_for_inpaint(vae, image, 1-mask, grow_mask_by=0)
    unet = apply_fooocus_inpaint(unet, inpaint_patch, latent)

    return unet

def sned_object_remove_request_to_api(
        checkpoint,
        prompt,
        num_inference_steps,
        guidance_scale,
        denoising_strength,
        seed,

        inpaint_model_name,
        inpaint_image,

        ip_addr
) :
    if isinstance(inpaint_image, NoneType):
        raise ValueError("inpaint_image가 None입니다. 올바른 이미지 객체를 전달하세요.")

    image, mask = inpaint_image['background'][:, :, :3], inpaint_image['layers'][0][:, :, 3]
    image = convert_image_to_base64(resize_image_for_sd(Image.fromarray(image)))
    mask = convert_image_to_base64(resize_image_for_sd(Image.fromarray(mask), is_mask=True))

    request_body = {
        'checkpoint': checkpoint,
        'inpaint_model_name': inpaint_model_name,
        'init_image': image,
        'mask': mask,
        "prompt_positive": prompt,
        "prompt_negative": "",
        'steps': num_inference_steps,
        'cfg': guidance_scale,
        'denoise': denoising_strength,
        'seed': seed,
    }

    url = f"http://{ip_addr}/object_remove"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    image = convert_base64_to_image_array(image_base64)
    return image

@torch.inference_mode()
def apply_fooocus_inpaint(unet, patch, latent) :
    module_path = 'ComfyUI/custom_nodes/comfyui-inpaint-nodes'
    func_name = 'nodes.ApplyFooocusInpaint'
    fooocus_applier = get_function_from_comfyui(module_path, func_name)
    unet = fooocus_applier().patch(unet, patch, latent)[0]
    return unet

@torch.inference_mode()
def webui_lama_proprecessor(image, mask) :
    mask = mask.unsqueeze(-1)
    image_rgba = torch.cat([image, mask], dim=-1).squeeze() * 255
    img = np.array(image_rgba).astype(np.uint8)
    H, W, C = img.shape
    assert C == 4, "No mask is provided!"
    raw_color = img[:, :, 0:3].copy()
    raw_mask = img[:, :, 3:4].copy()

    res = 256  # Always use 256 since lama is trained on 256

    img_res, remove_pad = resize_image_with_pad(img, res)

    model = load_lamaInpainting()
    # applied auto inversion
    prd_color = model(img_res)
    prd_color = remove_pad(prd_color)
    prd_color = cv2.resize(prd_color, (W, H))

    alpha = raw_mask.astype(np.float32) / 255.0
    fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(
        np.float32
    ) * (1 - alpha)
    fin_color = fin_color.clip(0, 255).astype(np.uint8)

    return torch.Tensor(fin_color)