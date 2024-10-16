import torch
from cgen_utils.loader import load_face_detailer, load_sam, load_detect_provider, load_dwpose_proprecessor, load_controlnet
import requests
from cgen_utils.handler import handle_response
from cgen_utils.image_process import (convert_base64_to_image_array,
                                      convert_image_array_to_base64,
                                      convert_image_tensor_to_base64,
                                      resize_image_for_sd,
                                      convert_image_to_base64,
                                      convert_base64_to_image_tensor,
                                      controlnet_image_preprocess
                                      )
from cgen_utils.comfyui import (apply_controlnet,
                                apply_ipadapter,
                                make_image_batch)
from cgen_utils.text_process import gemini_with_prompt_and_image
from cgen_utils.loader import load_clip_vision
from types import NoneType
from PIL import Image
import numpy as np
import torch.nn.functional as F


def sned_sam_request_to_api(
        image,
        prompt,
        threshold,
        ip_addr
) :
    if not isinstance(image, NoneType):
        image = resize_image_for_sd(Image.fromarray(image))
        image = convert_image_to_base64(image)

    request_body = {
        'sam_model': 'sam_vit_h_4b8939.pth',
        'dino_model': 'GroundingDINO_SwinB (938MB)',
        'image': image,
        'prompt': prompt,
        'threshold': threshold,
    }

    url = f"http://{ip_addr}/sam"
    response = requests.post(url, json=request_body)
    data = handle_response(response)
    image_base64 = data['image_base64']
    mask_base64 = data['mask_base64']
    mask_inv_base64 = data['mask_inv_base64']

    image = convert_base64_to_image_array(image_base64)
    mask = convert_base64_to_image_array(mask_base64)
    mask_inv = convert_base64_to_image_array(mask_inv_base64)

    return [image, mask, mask_inv]

